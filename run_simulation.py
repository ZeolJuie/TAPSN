import argparse
import logging
import time

import carla
import numpy as np
import tomli

from simulation.carla_simulation import CarlaSimulation
from simulation.obstacles import extract_sidewalk, extract_obstacles, get_dynamic_obstacles, extract_borders_from_config
from simulation.output_generator import OutputGenerator
from simulation.pedestrian_simulation import PedestrianSimulation
from simulation.pedestrian_spawner import PedSpawnManager
from sampling.collector import SamplingCollector
from sampling.config_loader import load_sampling_setup
from sampling.sequence_manager import SamplingSequenceManager
from simulation.vehicle_spawner import VehicleSpawnManager


class SimulationRunner:
    """
    SimulationRunner class is responsible for the synchronization of the Social-Force and CARLA
    simulations.
    """

    def __init__(self, pedestrian_simulation, carla_simulation, ped_spawn_manager, vehicle_spawn_manager,
                 scenario_config, args, sequence_manager=None, sampling_collector=None):

        self.ped_sim = pedestrian_simulation
        self.carla_sim = carla_simulation
        self.ped_spawn_manager = ped_spawn_manager
        self.vehicle_spawn_manager = vehicle_spawn_manager
        self.scenario_config = scenario_config
        self.sequence_manager = sequence_manager
        self.sampling_collector = sampling_collector

        self.output_csv = args.csv
        self.output_path = args.output
        self.duration = args.duration
        self.step_length = scenario_config.get('step_length', 0.05)

        walker_config = scenario_config.get('walker', {})
        self.draw_bounding_boxes = walker_config.get('draw_bounding_boxes', False)
        self.despawn_on_arrival = walker_config.get('despawn_on_arrival', True)
        self.waypoint_threshold = walker_config.get('waypoint_threshold', 2.0)

        self.walker_dict = ped_spawn_manager.walker_dict
        self.waypoint_dict = ped_spawn_manager.waypoint_dict
        self.scene_route_dict = ped_spawn_manager.scene_route_dict
        self.vehicle_list = vehicle_spawn_manager.vehicle_list
        self.vehicle_trajectory_dict = vehicle_spawn_manager.trajectory_dict
        self.vehicle_agent_dict = vehicle_spawn_manager.vehicle_agent_dict
        self.follow_ego_spectator = (
            self.sequence_manager is not None
            and self.sampling_collector is not None
        )
        self.ego_spectator_follow_distance = 4.5
        self.ego_spectator_follow_height = 2.5
        self.ego_spectator_follow_pitch = -15.0
        self._ego_spectator_follow_logged = False

    def _has_pending_ped_spawns(self):
        return any(spawner.quantity > 0 for spawner in self.ped_spawn_manager.ped_spawners)

    def _has_pending_vehicle_spawns(self):
        return any(spawner.quantity > 0 for spawner in self.vehicle_spawn_manager.vehicle_spawners)

    def _update_scene_route_navigation(self):
        if self.ped_sim.peds.state is None or not self.scene_route_dict:
            return set()

        completed_peds = set()
        active_ped_names = set(self.ped_sim.peds.name())
        for ped_name in [name for name in list(self.scene_route_dict.keys()) if name in active_ped_names]:
            ped = self.ped_sim.peds.state[self.ped_sim.peds.state['name'] == ped_name][0]
            follower = self.scene_route_dict[ped_name]
            navigation_target = follower.update(ped['loc'], ped['mode'].target_speed)
            self.ped_sim.peds.update_navigation_target(
                ped_name,
                navigation_target.next_waypoint,
                navigation_target.crossing_road,
            )
            self.ped_sim.update_scene_route_guidance(ped_name, navigation_target)
            if navigation_target.complete:
                completed_peds.add(ped_name)

        for ped_name in list(self.scene_route_dict.keys()):
            if ped_name not in active_ped_names:
                self.scene_route_dict.pop(ped_name, None)
                self.ped_sim.remove_scene_route_guidance(ped_name)

        return completed_peds

    def _despawn_pedestrian(self, ped_name):
        self.ped_sim.destroy_pedestrian(ped_name)
        self.carla_sim.destroy_actor(self.walker_dict[ped_name])
        self.walker_dict.pop(ped_name)
        self.ped_spawn_manager.walker_meta_dict.pop(ped_name, None)
        self.waypoint_dict.pop(ped_name, None)
        self.scene_route_dict.pop(ped_name, None)
        if self.sequence_manager is not None:
            self.sequence_manager.on_ped_despawned(ped_name)
        logging.info(f'Despawned pedestrian {ped_name}.')

    def is_complete(self):
        unresolved_ego_abort = self._unresolved_ego_abort_message()
        if unresolved_ego_abort is not None:
            raise RuntimeError(unresolved_ego_abort)

        if (
            self.sequence_manager is not None
            and self.sequence_manager.policy.get('stop_on_ego_end', True)
            and self.sequence_manager.resolved_ego is not None
            and self.sequence_manager.resolved_ego.get('despawned', False)
        ):
            logging.info('Stopping simulation because ego pedestrian completed and sampling.stop_on_ego_end=true.')
            return True

        if self.duration is not None and self.carla_sim.get_sim_time() >= self.duration:
            logging.info('Reached requested duration %.2fs. Stopping simulation.', self.duration)
            return True

        if self._has_pending_ped_spawns():
            return False

        if self.walker_dict:
            return False

        if self._has_pending_vehicle_spawns():
            return False

        if self.vehicle_list or self.vehicle_trajectory_dict or self.vehicle_agent_dict:
            return False

        return True

    def _unresolved_ego_abort_message(self):
        if self.sequence_manager is None:
            return None
        return self.ped_spawn_manager.unresolved_ego_abort_message(self.sequence_manager)

    def _update_ego_spectator_follow(self):
        if not self.follow_ego_spectator or self.sequence_manager is None:
            return

        resolved_ego = self.sequence_manager.resolved_ego
        if resolved_ego is None:
            return

        actor_id = int(resolved_ego.get('actor_id', 0))
        if actor_id <= 0:
            return

        ego_actor = self.carla_sim.world.get_actor(actor_id)
        if ego_actor is None or not ego_actor.is_alive:
            return

        ego_transform = ego_actor.get_transform()
        spectator_transform = carla.Transform(
            ego_transform.transform(
                carla.Vector3D(
                    x=-self.ego_spectator_follow_distance,
                    y=0.0,
                    z=self.ego_spectator_follow_height,
                )
            ),
            carla.Rotation(
                pitch=self.ego_spectator_follow_pitch,
                yaw=ego_transform.rotation.yaw,
                roll=0.0,
            ),
        )
        self.carla_sim.world.get_spectator().set_transform(spectator_transform)

        if not self._ego_spectator_follow_logged:
            logging.info(
                'Enabled spectator follow for ego actor_id=%d ped_name=%s.',
                actor_id,
                resolved_ego.get('ped_name'),
            )
            self._ego_spectator_follow_logged = True

    def tick(self):
        """
        Tick to simulation synchronization. One tick = one simulation step.
        """
        # spawn pedestrians and vehicles that are supposed to spawn in this time step
        sim_time = self.carla_sim.get_sim_time()
        self.ped_spawn_manager.tick(sim_time)
        self.vehicle_spawn_manager.tick(sim_time)
        if self.sampling_collector is not None:
            self.sampling_collector.try_activate()

        # Teleport vehicles that are not controlled by the traffic manager or agent to the next point on their
        # predefined trajectory
        for veh_id, values in list(self.vehicle_trajectory_dict.items()):
            if values['trajectory']:
                next_loc = values['trajectory'].pop(0)
                next_speed = values['speeds'].pop(0)
                self.carla_sim.update_actor(veh_id, next_loc, next_speed)
            elif not values['trajectory']:
                self.carla_sim.destroy_actor(veh_id)
                self.vehicle_trajectory_dict.pop(veh_id)
                self.vehicle_list.remove(veh_id)
                logging.info(f'Despawned vehicle {veh_id}.')

        # Apply control to all vehicles controlled by an agent
        for veh_id, agent in self.vehicle_agent_dict.items():
            if not agent.done():
                vehicle_control = agent.run_step()
                self.carla_sim.apply_vehicle_control(veh_id, vehicle_control)

        # Tick CARLA simulation and receive new location and velocity of all pedestrians and dynamic obstacles
        # (vehicles) and propagate the information to the pedestrian simulation
        if self.sampling_collector is not None:
            self.sampling_collector.update_sensor_world_poses()
        self.carla_sim.tick()
        snapshot = self.carla_sim.world.get_snapshot()
        self._update_ego_spectator_follow()
        for actor_id in self.walker_dict.values():
            walker = self.carla_sim.get_actor(actor_id)
            carla_location = walker.get_location()
            carla_velocity = walker.get_velocity()

            location = np.array([carla_location.x, carla_location.y, carla_location.z])
            velocity = np.array([carla_velocity.x, carla_velocity.y, carla_velocity.z])

            self.ped_sim.update_ped_info(actor_id, location, velocity)

            if self.draw_bounding_boxes:
                self.carla_sim.draw_bounding_box(actor_id, self.step_length)

        dynamic_obstacles, carla_borders = get_dynamic_obstacles(self.carla_sim.world, self.scenario_config)

        if dynamic_obstacles:
            self.ped_sim.update_dynamic_obstacles(dynamic_obstacles)

            if self.carla_sim.draw_obstacles:
                for border in carla_borders:
                    self.carla_sim.draw_points(border, self.step_length)

        if self.sampling_collector is not None:
            self.sampling_collector.capture_frame(snapshot)

        completed_scene_route_peds = self._update_scene_route_navigation()

        # Tick pedestrian simulation and propagate new velocities resulting from social force model to CARLA
        self.ped_sim.tick(sim_time)
        new_velocities = self.ped_sim.get_new_velocities()
        if new_velocities is not None:
            for velocity in new_velocities:
                walker_id = velocity['id'].item()
                new_velocity = velocity['vel']

                speed = np.linalg.norm(new_velocity)
                if speed != 0.0:
                    new_velocity = new_velocity / speed
                direction = carla.Vector3D(new_velocity[0], new_velocity[1], new_velocity[2])

                self.carla_sim.set_ped_velocity(walker_id, direction, speed)

        # get all pedestrians that arrived at their next waypoint and either assign a new waypoint or despawn
        # them if they reached their final destination
        scene_route_peds = set(self.scene_route_dict.keys())
        arrived_peds = set(self.ped_sim.get_arrived_peds(self.waypoint_threshold)) - scene_route_peds
        for ped_name in arrived_peds:
            remaining_waypoints = self.waypoint_dict[ped_name]

            if remaining_waypoints:
                next_waypoint = remaining_waypoints.pop(0)
                self.ped_sim.peds.update_next_waypoint(ped_name, next_waypoint)
                self.waypoint_dict[ped_name] = remaining_waypoints

            elif not remaining_waypoints and self.despawn_on_arrival:
                self._despawn_pedestrian(ped_name)

        if self.despawn_on_arrival:
            for ped_name in completed_scene_route_peds:
                if ped_name in self.walker_dict:
                    self._despawn_pedestrian(ped_name)

    def close(self):
        """
        Cleans synchronization.
        """
        # Destroying walkers.
        for carla_actor_id in self.walker_dict.values():
            self.carla_sim.destroy_actor(carla_actor_id)

        # Destroying vehicles.
        for carla_actor_id in self.vehicle_list:
            self.carla_sim.destroy_actor(carla_actor_id)

        # Closing pedestrian simulation and CARLA client.
        if self.sampling_collector is not None:
            self.sampling_collector.close()
        self.carla_sim.close()
        self.ped_sim.close()

        should_generate_simulation_records = self.output_csv or (
            self.sequence_manager is not None
            and self.sequence_manager.policy.get('export', {}).get('simulation_records', True)
        )

        if should_generate_simulation_records:
            scenario_name = self.scenario_config.get('scenario_name')
            sequence_simulation_dir = (
                self.sequence_manager.paths['simulation'] if self.sequence_manager is not None else None
            )

            output_gen = OutputGenerator(
                self.ped_sim,
                self.output_path,
                scenario_name,
                weather_metadata=self.carla_sim.generated_weather,
                output_dir=sequence_simulation_dir,
            )
            output_gen.generate_ped_csv()
            output_gen.generate_veh_csv()
            output_gen.generate_borders_csv()
            output_gen.generate_obstacles_csv()
            output_gen.generate_weather_json()
            if self.sequence_manager is not None:
                self.sequence_manager.finalize_from_sim_output(None, self.carla_sim)


def simulation_loop(args):
    """
    Entry point for CARLA simulation with social force pedestrian model. Main simulation loop.
    """

    # load configs
    scenario_config = load_config(args.scenario_config)
    sfm_config = load_config(args.sfm_config)
    step_length = scenario_config.get('step_length', 0.05)
    sampling_setup = load_sampling_setup(scenario_config, args.scenario_config)
    sequence_manager = SamplingSequenceManager(scenario_config, sampling_setup) if sampling_setup else None

    logging.info('Stage: init CarlaSimulation start')
    carla_simulation = CarlaSimulation(args, scenario_config)
    logging.info('Stage: init CarlaSimulation done map=%s', carla_simulation.carla_map.name)

    logging.info('Stage: extract borders from scenario config start')
    borders, section_info, carla_borders = extract_borders_from_config(scenario_config)
    logging.info(
        'Stage: extract borders from scenario config done borders=%d sections=%d carla_borders=%d',
        len(borders),
        len(section_info),
        len(carla_borders),
    )

    logging.info('Stage: extract sidewalk borders start')
    sidewalk_borders, sidewalk_section_info, carla_sidewalk_borders = extract_sidewalk(carla_simulation.carla_map,
                                                                                       scenario_config)
    logging.info(
        'Stage: extract sidewalk borders done sidewalks=%d sidewalk_sections=%d carla_sidewalk_borders=%d',
        len(sidewalk_borders),
        len(sidewalk_section_info),
        len(carla_sidewalk_borders),
    )
    borders.extend(sidewalk_borders)
    section_info.extend(sidewalk_section_info)
    carla_borders.extend(carla_sidewalk_borders)
    logging.info(
        'Stage: merged manual+sidewalk borders done total_borders=%d total_sections=%d total_carla_borders=%d',
        len(borders),
        len(section_info),
        len(carla_borders),
    )

    logging.info('Stage: extract obstacles from map start')
    obstacle_positions, obstacle_borders, carla_obstacle_borders = extract_obstacles(carla_simulation.world,
                                                                                     scenario_config)
    logging.info(
        'Stage: extract obstacles from map done obstacle_positions=%d obstacle_borders=%d carla_obstacle_borders=%d',
        len(obstacle_positions),
        len(obstacle_borders),
        len(carla_obstacle_borders),
    )

    carla_borders.extend(carla_obstacle_borders)
    obstacles = list(zip(obstacle_positions, obstacle_borders))

    # draw obstacles
    if carla_simulation.draw_obstacles:
        for border in carla_borders:
            carla_simulation.draw_points(border, 30)

    # initialize pedestrian simulation
    pedestrian_simulation = PedestrianSimulation(borders, section_info, obstacles, sfm_config, step_length)

    # initialize pedestrian and vehicle spawn manager
    ped_spawn_manager = PedSpawnManager(
        scenario_config,
        carla_simulation,
        pedestrian_simulation,
        spawn_event_listener=sequence_manager,
    )
    vehicle_spawn_manager = VehicleSpawnManager(scenario_config, carla_simulation)
    sampling_collector = SamplingCollector(carla_simulation, sequence_manager) if sequence_manager else None

    # initialize simulation runner
    sim_runner = SimulationRunner(
        pedestrian_simulation,
        carla_simulation,
        ped_spawn_manager,
        vehicle_spawn_manager,
        scenario_config,
        args,
        sequence_manager=sequence_manager,
        sampling_collector=sampling_collector,
    )

    # main simulation loop
    try:
        while True:
            start = time.time()

            sim_runner.tick()
            if sim_runner.is_complete():
                logging.info('Scenario completed. No active actors or pending spawns remain.')
                break

            end = time.time()
            elapsed = end - start
            if elapsed < step_length:
                time.sleep(step_length - elapsed)

    except KeyboardInterrupt:
        logging.info('Cancelled by user.')

    finally:
        logging.info('Cleaning Simulation')

        sim_runner.close()


def load_config(config_path):
    """
    Loads configuration from file.
    :param config_path:
    :return:
    """
    with open(config_path, mode='rb') as fp:
        config = tomli.load(fp)
    return config


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--scenario-config',
                           required=True,
                           type=str,
                           help='scenario configuration file')
    argparser.add_argument('--sfm-config',
                           default='config/sfm_config.toml',
                           type=str,
                           help='social force model configuration file')
    argparser.add_argument('--carla-host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the carla_sim host server (default: 127.0.0.1)')
    argparser.add_argument('--carla-port',
                           metavar='P',
                           default=2000,
                           type=int,
                           help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--csv', action='store_true', help='output csv with sim results')
    argparser.add_argument('--output',
                           default='output',
                           type=str,
                           help='path for output CSV files')
    argparser.add_argument('--duration',
                           default=None,
                           type=float,
                           help='optional maximum simulated time in seconds before stopping gracefully')
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    arguments = argparser.parse_args()

    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    simulation_loop(arguments)

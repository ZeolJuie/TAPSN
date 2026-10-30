import logging
import math
import random

import carla
import numpy as np

from agents.navigation.behavior_agent import BehaviorAgent
from .scene_catalog import SceneCatalog, SceneCatalogError


class VehicleSpawnManager:
    """
    Class managing all vehicle spawners.
    """

    def __init__(self, scenario_config, carla_sim):
        self.scenario_config = scenario_config
        self.carla_sim = carla_sim

        self.vehicle_seed = scenario_config.get('vehicle', {}).get('vehicle_seed', 2000)
        self.variate_speed_factor = scenario_config.get('vehicle', {}).get('variate_speed_factor', 0.0)
        self.recommended_spawn_points = carla_sim.world.get_map().get_spawn_points()
        no_bikes = scenario_config.get('vehicle', {}).get('no_bikes', False)
        self.background_population_config = scenario_config.get('vehicle', {}).get('background_population', {})

        blueprint_library = carla_sim.world.get_blueprint_library().filter('vehicle')
        if no_bikes:
            self.vehicle_blueprints = [x for x in blueprint_library if int(x.get_attribute('number_of_wheels')) == 4]
        else:
            self.vehicle_blueprints = blueprint_library

        # get vehicle spawners from scenario config
        self.vehicle_spawners = self._extract_vehicle_info()
        self.vehicle_spawners.extend(self._extract_background_population_info())

        self.SpawnActor = carla.command.SpawnActor
        self.SetAutopilot = carla.command.SetAutopilot
        self.FutureActor = carla.command.FutureActor

        self.traffic_manager = None

        self.vehicle_list = []
        self.trajectory_dict = {}
        self.vehicle_agent_dict = {}

    def _ensure_traffic_manager(self):
        if self.traffic_manager is None:
            logging.info(
                'TM init: begin map=%s seed=%d port=%d',
                self.scenario_config.get('map', {}).get('map_name'),
                self.vehicle_seed,
                8000,
            )
            logging.info('TM init: calling client.get_trafficmanager(%d)', 8000)
            self.traffic_manager = self.carla_sim.client.get_trafficmanager(8000)
            logging.info('TM init: client.get_trafficmanager(%d) ok', 8000)
            logging.info('TM init: calling traffic_manager.set_synchronous_mode(True)')
            self.traffic_manager.set_synchronous_mode(True)
            logging.info('TM init: traffic_manager.set_synchronous_mode(True) ok')
            logging.info('TM init: calling traffic_manager.set_random_device_seed(%d)', self.vehicle_seed)
            self.traffic_manager.set_random_device_seed(self.vehicle_seed)
            logging.info('TM init: traffic_manager.set_random_device_seed(%d) ok', self.vehicle_seed)
        return self.traffic_manager

    def tick(self, sim_time):
        """
        Check if vehicle spawners are ready to spawn in this simulation step and let them spawn vehicles
        if that's the case.
        :param sim_time:
        :return:
        """
        # filter out vehicle spawner that don't have any vehicles left to spawn
        self.vehicle_spawners[:] = [spawner for spawner in self.vehicle_spawners if spawner.quantity > 0]

        for vehicle_spawner in self.vehicle_spawners:
            if vehicle_spawner.ready_to_spawn(sim_time):
                self._spawn_vehicle(vehicle_spawner)
                vehicle_spawner.quantity -= 1

    def _extract_vehicle_info(self):
        """
        Extracts vehicle spawner information from the configuration file.
        """

        vehicle_spawner_config = self.scenario_config.get('vehicle', {}).get('vehicle_spawner')

        vehicle_spawners = []
        if vehicle_spawner_config is not None:
            for spawner in vehicle_spawner_config:
                spawn_location = spawner.get('spawn_point')
                blueprint = spawner.get('blueprint')
                auto_pilot = spawner.get('auto_pilot', True)
                use_traffic_manager = spawner.get('use_traffic_manager', True)
                destination = spawner.get('destination')
                trajectory = spawner.get('trajectory', [])
                headings = spawner.get('headings', [])
                speeds = spawner.get('speeds', [])
                speed_reduction_factor = spawner.get('speed_reduction_factor', 30)
                quantity = spawner.get('quantity', 1)
                spawn_time = spawner.get('spawn_time', 0.0)
                spawn_interval = spawner.get('spawn_interval', 5.0)
                ignore_walkers_percentage = spawner.get('ignore_walkers_percentage', 0)
                ignore_lights_percentage = spawner.get('ignore_lights_percentage', 0)

                vehicle_spawner = VehicleSpawner(spawn_location, blueprint, auto_pilot, use_traffic_manager,
                                                 destination, trajectory, headings, speeds,
                                                 speed_reduction_factor, quantity, spawn_time, spawn_interval,
                                                 ignore_walkers_percentage, ignore_lights_percentage,
                                                 self.recommended_spawn_points)
                vehicle_spawners.append(vehicle_spawner)

        return vehicle_spawners

    def _extract_background_population_info(self):
        config = self.background_population_config
        if not config or not config.get('enabled', False):
            return []

        local_count = int(config.get('local_count', 0))
        global_count = int(config.get('global_count', 0))
        if local_count < 0 or global_count < 0:
            raise ValueError('vehicle.background_population local_count/global_count must be non-negative.')
        if local_count == 0 and global_count == 0:
            return []

        population_seed = int(config.get('seed', self.vehicle_seed))
        rng = random.Random(population_seed)
        local_candidates, global_candidates = self._background_population_candidate_indices(local_count)

        selected_local = _sample_without_replacement(rng, local_candidates, local_count)
        remaining_indices = [idx for idx in range(len(self.recommended_spawn_points)) if idx not in set(selected_local)]
        preferred_global = [idx for idx in global_candidates if idx in remaining_indices]
        fallback_global = [idx for idx in remaining_indices if idx not in set(preferred_global)]
        selected_global = _sample_without_replacement(rng, preferred_global, global_count)
        if len(selected_global) < global_count:
            selected_global.extend(
                _sample_without_replacement(
                    rng,
                    fallback_global,
                    global_count - len(selected_global),
                )
            )

        selected_indices = [('local', idx) for idx in selected_local] + [('global', idx) for idx in selected_global]
        if not selected_indices:
            logging.warning('vehicle.background_population enabled, but no spawn points were selected.')
            return []

        speed_reduction_mean = float(config.get('speed_reduction_mean', 10.0))
        speed_reduction_jitter = max(0.0, float(config.get('speed_reduction_jitter', 5.0)))
        auto_pilot = config.get('auto_pilot', True)
        # use_traffic_manager = config.get('use_traffic_manager', True)
        use_traffic_manager = False
        ignore_walkers_percentage = float(config.get('ignore_walkers_percentage', 0))
        ignore_lights_percentage = float(config.get('ignore_lights_percentage', 0))
        spawn_time = float(config.get('spawn_time', 0.0))
        spawn_interval = float(config.get('spawn_interval', 0.0))

        requested_count = local_count + global_count
        actual_count = len(selected_indices)
        if actual_count < requested_count:
            logging.warning(
                'vehicle.background_population requested %d vehicles, but only selected %d unique spawn points.',
                requested_count,
                actual_count,
            )

        spawners = []
        for population_bucket, spawn_point_idx in selected_indices:
            blueprint = _resolve_background_blueprint(config, rng)
            speed_reduction_factor = speed_reduction_mean + rng.uniform(-speed_reduction_jitter, speed_reduction_jitter)
            vehicle_spawner = VehicleSpawner(
                spawn_point=spawn_point_idx,
                blueprint=blueprint,
                auto_pilot=auto_pilot,
                use_traffic_manager=use_traffic_manager,
                destination=None,
                trajectory=[],
                headings=[],
                speeds=[],
                speed_reduction_factor=speed_reduction_factor,
                quantity=1,
                spawn_time=spawn_time,
                spawn_interval=spawn_interval,
                ignore_walkers_percentage=ignore_walkers_percentage,
                ignore_lights_percentage=ignore_lights_percentage,
                recommended_spawn_points=self.recommended_spawn_points,
            )
            vehicle_spawner.population_bucket = population_bucket
            spawners.append(vehicle_spawner)

        logging.info(
            'Prepared background vehicle population: local=%d global=%d seed=%d scene_id=%s.',
            len(selected_local),
            len(selected_global),
            population_seed,
            self.scenario_config.get('map', {}).get('scene_id'),
        )
        return spawners

    def _background_population_candidate_indices(self, local_count):
        spawn_count = len(self.recommended_spawn_points)
        if spawn_count == 0:
            return [], []

        if local_count <= 0:
            return [], list(range(spawn_count))

        scene_bounds = _load_scene_bounds(self.scenario_config.get('map', {}))
        if scene_bounds is None:
            logging.warning(
                'vehicle.background_population local_count > 0 but scene bounds could not be resolved. '
                'Falling back to global-only spawn selection.'
            )
            return [], list(range(spawn_count))

        distance_rank = []
        for index, transform in enumerate(self.recommended_spawn_points):
            location = transform.location
            distance_rank.append(
                (
                    _distance_to_bounds(location.x, location.y, scene_bounds),
                    index,
                )
            )
        distance_rank.sort(key=lambda item: (item[0], item[1]))

        local_pool_size = min(spawn_count, max(local_count * 6, 24))
        local_candidates = [index for _, index in distance_rank[:local_pool_size]]
        global_candidates = [index for _, index in distance_rank[local_pool_size:]]
        return local_candidates, global_candidates

    def _spawn_vehicle(self, vehicle_spawner):
        """
        Spawn vehicle in Carla simulation.
        """
        spawn_transform = vehicle_spawner.carla_spawn_transform

        random.seed(self.vehicle_seed)

        if vehicle_spawner.blueprint:
            vehicle_bp = self.vehicle_blueprints.find(vehicle_spawner.blueprint)
        else:
            vehicle_bp = random.choice(self.vehicle_blueprints)

        # spawn vehicle in Carla
        tm_autopilot = vehicle_spawner.auto_pilot and vehicle_spawner.use_traffic_manager
        logging.info(
            'Vehicle control path: spawn_point=%s bucket=%s auto_pilot=%s use_traffic_manager=%s tm_autopilot=%s',
            vehicle_spawner.spawn_point_index,
            getattr(vehicle_spawner, 'population_bucket', None),
            vehicle_spawner.auto_pilot,
            vehicle_spawner.use_traffic_manager,
            tm_autopilot,
        )
        traffic_manager = self._ensure_traffic_manager() if tm_autopilot else None
        spawn_location = spawn_transform.location
        logging.info(
            'Preparing vehicle spawn blueprint=%s spawn_point=%s bucket=%s location=[%.3f, %.3f, %.3f] '
            'auto_pilot=%s use_traffic_manager=%s tm_autopilot=%s',
            vehicle_bp.id,
            vehicle_spawner.spawn_point_index,
            getattr(vehicle_spawner, 'population_bucket', None),
            float(spawn_location.x),
            float(spawn_location.y),
            float(spawn_location.z),
            vehicle_spawner.auto_pilot,
            vehicle_spawner.use_traffic_manager,
            tm_autopilot,
        )
        if tm_autopilot:
            logging.info(
                'TM spawn: building SpawnActor+SetAutopilot batch spawn_point=%s tm_port=%s',
                vehicle_spawner.spawn_point_index,
                traffic_manager.get_port(),
            )
            batch = [self.SpawnActor(vehicle_bp, spawn_transform)
                         .then(self.SetAutopilot(self.FutureActor, True, traffic_manager.get_port()))]
            logging.info(
                'TM spawn: SpawnActor+SetAutopilot batch ready spawn_point=%s',
                vehicle_spawner.spawn_point_index,
            )
        else:
            batch = [self.SpawnActor(vehicle_bp, spawn_transform)]
            logging.info(
                'Vehicle spawn: SpawnActor-only batch ready spawn_point=%s',
                vehicle_spawner.spawn_point_index,
            )
        logging.info(
            'Vehicle spawn: calling apply_batch_sync spawn_point=%s tm_autopilot=%s',
            vehicle_spawner.spawn_point_index,
            tm_autopilot,
        )
        actor_id = self.carla_sim.spawn_actor_with_batch(batch)
        logging.info(
            'Vehicle spawn: apply_batch_sync returned actor_id=%s spawn_point=%s',
            actor_id,
            vehicle_spawner.spawn_point_index,
        )

        if self.variate_speed_factor != 0.0:
            vehicle_spawner.speed_reduction_factor += random.uniform(-self.variate_speed_factor,
                                                                     self.variate_speed_factor)

        # increment vehicle seed so next walker has a different random blueprint and speed variation
        self.vehicle_seed += 1

        if actor_id != -1:

            vehicle = self.carla_sim.world.get_actor(actor_id)
            self.vehicle_list.append(actor_id)

            if vehicle_spawner.auto_pilot:
                if vehicle_spawner.use_traffic_manager:
                    logging.info(
                        'TM post-spawn: configuring actor_id=%s speed_diff=%.3f ignore_walkers=%.3f ignore_lights=%.3f',
                        actor_id,
                        vehicle_spawner.speed_reduction_factor,
                        vehicle_spawner.ignore_walkers_percentage,
                        vehicle_spawner.ignore_lights_percentage,
                    )
                    traffic_manager.vehicle_percentage_speed_difference(vehicle,
                                                                       vehicle_spawner.speed_reduction_factor)
                    logging.info('TM post-spawn: vehicle_percentage_speed_difference ok actor_id=%s', actor_id)
                    traffic_manager.ignore_walkers_percentage(vehicle, vehicle_spawner.ignore_walkers_percentage)
                    logging.info('TM post-spawn: ignore_walkers_percentage ok actor_id=%s', actor_id)
                    traffic_manager.ignore_lights_percentage(vehicle, vehicle_spawner.ignore_lights_percentage)
                    logging.info('TM post-spawn: ignore_lights_percentage ok actor_id=%s', actor_id)
                else:
                    self.carla_sim.tick()
                    agent = BehaviorAgent(vehicle)
                    if vehicle_spawner.destination is not None:
                        agent.set_destination(vehicle_spawner.carla_destination_transform.location,
                                              vehicle_spawner.carla_spawn_transform.location)
                    agent.ignore_traffic_lights(vehicle_spawner.ignore_lights_percentage > 0)
                    self.vehicle_agent_dict[actor_id] = agent
            else:
                carla_trajectory = [generate_carla_transform(location, heading)
                                    for location, heading in zip(vehicle_spawner.trajectory, vehicle_spawner.headings)]
                self.trajectory_dict[actor_id] = {}
                self.trajectory_dict[actor_id]['trajectory'] = carla_trajectory
                self.trajectory_dict[actor_id]['speeds'] = vehicle_spawner.speeds

            logging.info(f'Spawned vehicle {actor_id} of type {vehicle.type_id}.')


class VehicleSpawner:
    """
    Class containing all the information necessary to spawn one or multiple vehicles from one spawn point.
    """

    def __init__(self, spawn_point, blueprint, auto_pilot, use_traffic_manager, destination, trajectory, headings,
                 speeds, speed_reduction_factor, quantity, spawn_time, spawn_interval, ignore_walkers_percentage,
                 ignore_lights_percentage, recommended_spawn_points):
        self.spawn_point = spawn_point
        self.blueprint = blueprint
        self.auto_pilot = auto_pilot
        self.use_traffic_manager = use_traffic_manager
        self.destination = destination
        self.trajectory = trajectory
        self.headings = headings
        self.speeds = speeds[1:]
        self.speed_reduction_factor = speed_reduction_factor
        self.quantity = quantity
        self.spawn_interval = spawn_interval
        self.next_spawn_time = spawn_time
        self.ignore_walkers_percentage = ignore_walkers_percentage
        self.ignore_lights_percentage = ignore_lights_percentage
        self.recommended_spawn_points = recommended_spawn_points
        self.population_bucket = None
        self.spawn_point_index = spawn_point

        self.carla_spawn_transform = self.generate_carla_spawn_transform()
        if self.destination is not None:
            self.carla_destination_transform = self.recommended_spawn_points[self.destination]

    def ready_to_spawn(self, sim_time):
        """
        Check if vehicle spawner is ready to spawn in this simulation time step.
        :param sim_time:
        :return: True if vehicle spawner is ready to spawn and False if not
        """
        if self.next_spawn_time <= sim_time:
            self.next_spawn_time += self.spawn_interval
            return True
        else:
            return False

    def generate_carla_spawn_transform(self):
        """
        Generate spawn transform for Carla simulator.
        :return: Carla spawn point
        """
        if self.spawn_point is not None:
            transform = self.recommended_spawn_points[self.spawn_point]
        else:
            spawn_loc = self.trajectory.pop(0)
            heading = self.headings.pop(0)

            transform = generate_carla_transform(spawn_loc, heading, 1.0)

        return transform


def generate_carla_transform(location, heading, z_loc=0.0):
    transform = carla.Transform()
    transform.location = carla.Location(location[0], location[1], z_loc)
    transform.rotation = carla.Rotation(0, np.degrees(heading), 0)

    return transform


def _sample_without_replacement(rng, candidates, count):
    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return shuffled
    return rng.sample(list(candidates), count)


def _resolve_background_blueprint(config, rng):
    blueprints = config.get('blueprints')
    if blueprints:
        return rng.choice(list(blueprints))
    return config.get('blueprint')


def _load_scene_bounds(map_config):
    scene_id = map_config.get('scene_id')
    if not scene_id:
        return None
    try:
        scene = SceneCatalog.from_map_config(map_config).get_scene(str(scene_id))
    except SceneCatalogError:
        return None
    return scene.get('bounds')


def _distance_to_bounds(x, y, bounds):
    min_x = float(bounds['min_x'])
    max_x = float(bounds['max_x'])
    min_y = float(bounds['min_y'])
    max_y = float(bounds['max_y'])
    dx = max(min_x - x, 0.0, x - max_x)
    dy = max(min_y - y, 0.0, y - max_y)
    return math.hypot(dx, dy)

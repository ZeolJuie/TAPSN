import json
import logging
import math
import time
from pathlib import Path

import carla

from .weather_generator import generate_weather


DEFAULT_TILE_STREAM_DISTANCE_M = 1200.0
DEFAULT_ACTOR_ACTIVE_DISTANCE_M = 1200.0
DEFAULT_TILE_SPECTATOR_HEIGHT_M = 220.0
DEFAULT_TILE_SPECTATOR_PITCH_DEG = -88.0
DEFAULT_TILE_STREAMING_WAIT_TICKS = 8
DEFAULT_TILE_STREAMING_WAIT_TIMEOUT_S = 1.0


class CarlaSimulation:

    def __init__(self, args, config):

        self.config = config
        self.map_config = self.config['map']
        self.map_name = self.map_config['map_name']
        self.map_path = self.map_config['map_path']
        self.unload_props = self.map_config.get('unload_props', False)
        self.draw_obstacles = self.map_config.get('draw_obstacles', False)
        self.carla_timeout = self.map_config.get('carla_timeout', 10.0)
        self.generated_weather = None
        self.tile_streaming_context = self._load_tile_streaming_context()

        # connect to CARLA server
        self.client = carla.Client(args.carla_host, args.carla_port)
        self.client.set_timeout(self.carla_timeout)
        self.world = self.client.get_world()
        self.start_time = self.world.get_snapshot().timestamp.elapsed_seconds

        # load configured map
        self.carla_map = self.world.get_map()
        if self.carla_map.name != self.map_path + self.map_name:
            self.world = self.client.load_world(self.map_name)
            # update map variable and start time after loading new map
            self.carla_map = self.world.get_map()

        self._apply_tile_streaming_anchor_if_needed()

        weather_config = self.config.get('weather')
        if weather_config is not None:
            fallback_weather_seed = self.config.get('walker', {}).get('pedestrian_seed', 0)
            weather, weather_metadata = generate_weather(weather_config, fallback_weather_seed)
            if weather is not None:
                self.world.set_weather(weather)
                self.generated_weather = weather_metadata

        # unload props
        if self.unload_props:
            self.world.unload_map_layer(carla.MapLayer.Props)
            self.world.unload_map_layer(carla.MapLayer.StreetLights)
            self.world.unload_map_layer(carla.MapLayer.Walls)
            self.world.unload_map_layer(carla.MapLayer.Foliage)

        # Configuring CARLA simulation in sync mode.
        self.original_settings = self.world.get_settings()
        self.step_length = self.config.get('step_length', 0.05)
        self.sub_step_length = self.config.get('sub_step_length', -1)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.deterministic_ragdolls = True
        settings.fixed_delta_seconds = self.step_length

        if self.sub_step_length > 0:
            settings.substepping = True
            settings.max_substep_delta_time = self.sub_step_length
            settings.max_substeps = math.ceil(self.step_length / self.sub_step_length)

        self.world.apply_settings(settings)
        self.start_time = self.world.get_snapshot().timestamp.elapsed_seconds

        # set spectator
        spectator_loc = self.map_config.get('spectator_location')
        spectator_rot = self.map_config.get('spectator_rotation')

        if self.tile_streaming_context is None and spectator_loc is not None and spectator_rot is not None:
            spectator = self.world.get_spectator()
            spectator_transform = carla.Transform()
            spectator_transform.location = carla.Location(spectator_loc[0], spectator_loc[1], spectator_loc[2])
            spectator_transform.rotation = carla.Rotation(spectator_rot[0], spectator_rot[1], spectator_rot[2])
            spectator.set_transform(spectator_transform)

        self.blueprint_library = self.world.get_blueprint_library()
        self.walker_blueprints = self.blueprint_library.filter('walker.pedestrian.*')

    def _load_tile_streaming_context(self):
        scene_catalog_path = self.map_config.get('scene_catalog_path')
        if not scene_catalog_path:
            return None

        catalog_path = Path(__file__).resolve().parents[1] / Path(scene_catalog_path)
        if not catalog_path.exists():
            logging.warning('Tile streaming anchor skipped because scene catalog does not exist: %s', catalog_path)
            return None

        try:
            payload = json.loads(catalog_path.read_text(encoding='utf-8'))
        except Exception as exc:
            logging.warning('Tile streaming anchor skipped because scene catalog could not be parsed: %s', exc)
            return None

        tile = payload.get('tile')
        if not isinstance(tile, dict):
            return None

        core_bounds = tile.get('core_bounds')
        halo_bounds = tile.get('halo_bounds', core_bounds)
        if not core_bounds or not halo_bounds:
            logging.warning('Tile streaming anchor skipped because tile bounds are incomplete in %s', catalog_path)
            return None

        halo_min_x, halo_min_y, halo_max_x, halo_max_y = [float(value) for value in halo_bounds]
        halo_width = halo_max_x - halo_min_x
        halo_height = halo_max_y - halo_min_y
        stream_distance = max(DEFAULT_TILE_STREAM_DISTANCE_M, max(halo_width, halo_height) + 400.0)
        actor_active_distance = max(DEFAULT_ACTOR_ACTIVE_DISTANCE_M, stream_distance)

        return {
            'catalog_path': str(catalog_path),
            'tile_id': tile.get('id'),
            'core_bounds': [float(value) for value in core_bounds],
            'halo_bounds': [float(value) for value in halo_bounds],
            'stream_distance': float(stream_distance),
            'actor_active_distance': float(actor_active_distance),
        }

    @staticmethod
    def _bounds_center(bounds):
        min_x, min_y, max_x, max_y = [float(value) for value in bounds]
        return carla.Location(x=(min_x + max_x) * 0.5, y=(min_y + max_y) * 0.5, z=0.0)

    def _nearest_waypoint_height(self, center):
        for lane_type in (carla.LaneType.Sidewalk, carla.LaneType.Driving, carla.LaneType.Any):
            try:
                waypoint = self.carla_map.get_waypoint(center, lane_type=lane_type)
            except RuntimeError:
                waypoint = None
            if waypoint is not None:
                return float(waypoint.transform.location.z)
        return 0.0

    def _apply_tile_streaming_anchor_if_needed(self):
        if self.tile_streaming_context is None:
            return

        settings = self.world.get_settings()
        if hasattr(settings, 'tile_stream_distance'):
            settings.tile_stream_distance = self.tile_streaming_context['stream_distance']
        if hasattr(settings, 'actor_active_distance'):
            settings.actor_active_distance = self.tile_streaming_context['actor_active_distance']
        if hasattr(settings, 'spectator_as_ego'):
            settings.spectator_as_ego = True
        self.world.apply_settings(settings)

        center = self._bounds_center(self.tile_streaming_context['core_bounds'])
        base_z = self._nearest_waypoint_height(center)
        spectator_transform = carla.Transform(
            carla.Location(x=center.x, y=center.y, z=base_z + DEFAULT_TILE_SPECTATOR_HEIGHT_M),
            carla.Rotation(pitch=DEFAULT_TILE_SPECTATOR_PITCH_DEG, yaw=0.0, roll=0.0),
        )
        self.world.get_spectator().set_transform(spectator_transform)
        logging.info(
            'Tile streaming anchor enabled tile_id=%s center=[%.3f, %.3f, %.3f] stream_distance=%.1f actor_active_distance=%.1f catalog=%s',
            self.tile_streaming_context['tile_id'],
            center.x,
            center.y,
            base_z,
            self.tile_streaming_context['stream_distance'],
            self.tile_streaming_context['actor_active_distance'],
            self.tile_streaming_context['catalog_path'],
        )

        for tick_index in range(DEFAULT_TILE_STREAMING_WAIT_TICKS):
            try:
                self.world.wait_for_tick(DEFAULT_TILE_STREAMING_WAIT_TIMEOUT_S)
            except RuntimeError:
                time.sleep(0.2)
            logging.info(
                'Tile streaming anchor warmup tick=%d/%d tile_id=%s',
                tick_index + 1,
                DEFAULT_TILE_STREAMING_WAIT_TICKS,
                self.tile_streaming_context['tile_id'],
            )

    def tick(self):
        """
        Tick to CARLA simulation.
        """
        self.world.tick()

    def get_actor(self, actor_id):
        """
        Accessor for CARLA actor.
        """
        return self.world.get_actor(actor_id)

    def spawn_actor(self, blueprint, transform):
        """
        Spawns a new actor.
        :param blueprint: blueprint of the actor to be spawned.
        :param transform: transform where the actor will be spawned.
        :return: actor id if the actor is successfully spawned. Otherwise, INVALID_ACTOR_ID.
        """
        transform = carla.Transform(transform.location, transform.rotation)
        batch = [carla.command.SpawnActor(blueprint, transform)]

        return self.spawn_actor_with_batch(batch)

    def spawn_actor_with_batch(self, batch):
        """
        Spawns a new actor with batch command.
        :param batch: batch of carla commands
        :return: actor id if the actor is successfully spawned. Otherwise, INVALID_ACTOR_ID.
        """
        response = self.client.apply_batch_sync(batch, False)[0]
        if response.error:
            logging.error('Spawn carla actor failed. %s', response.error)
            return -1

        return response.actor_id

    def update_actor(self, actor_id, transform, speed=0):
        actor = self.world.get_actor(actor_id)
        velocity = transform.get_forward_vector() * speed
        actor.set_transform(transform)
        actor.set_target_velocity(velocity)

    def destroy_actor(self, actor_id):
        """
        Destroys the given actor.
        """
        actor = self.world.get_actor(actor_id)
        if actor is not None:
            return actor.destroy()
        return False

    def apply_vehicle_control(self, actor_id, vehicle_control):
        vehicle = self.world.get_actor(actor_id)
        vehicle.apply_control(vehicle_control)

    def set_ped_velocity(self, walker_id, direction, speed):
        walker = self.world.get_actor(walker_id)
        walker_control = carla.WalkerControl(direction, speed, False)
        walker.apply_control(walker_control)

    def get_ped_transform(self, walker_id):
        walker = self.world.get_actor(walker_id)
        transform = walker.get_transform()

        return transform

    def get_ped_radius(self, walker_id):
        walker = self.world.get_actor(walker_id)
        extent = walker.bounding_box.extent
        radius = max([extent.x, extent.y])
        return radius

    def get_sim_time(self):
        timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
        sim_time = timestamp - self.start_time
        return sim_time

    def draw_lines(self, lines):
        for line in lines:
            self.world.debug.draw_line(line[0], line[1], color=carla.Color(0, 0, 0, 0), thickness=0.05, life_time=0)

    def draw_bounding_box(self, actor_id, step_length):
        actor = self.world.get_actor(actor_id)
        bb = carla.BoundingBox(actor.get_location(), actor.bounding_box.extent)
        self.world.debug.draw_box(bb, actor.get_transform().rotation, color=carla.Color(0, 0, 0, 0), thickness=0.01,
                                  life_time=step_length + 0.00000001)

    def draw_points(self, points, lt=0):
        for point in points:
            self.world.debug.draw_point(point, size=0.05, life_time=lt + 0.00000001)

    def close(self):
        # reset CARLA simulation settings
        self.world.apply_settings(self.original_settings)

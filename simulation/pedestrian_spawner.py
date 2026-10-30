import logging
import random

import carla
import numpy as np

from . import stateutils
from .path_planner import PedPathPlanner, GraphType
from .ped_mode_manager import PedMode, PedModeManager
from .scene_route_follower import SceneRoute, SceneRouteFollower
from .scene_catalog import SceneCatalog, SceneCatalogError

AUTO_SPAWN_Z_LIFTS = (0.0, 0.2)
AUTO_SPAWN_OFFSET_PAIRS = (
    (0.0, 0.0),
    (0.0, 0.5),
    (0.0, -0.5),
    (-0.5, 0.0),
    (0.5, 0.0),
    (-0.5, 0.5),
    (-0.5, -0.5),
    (0.5, 0.5),
    (0.5, -0.5),
)
ROUTE_SPAWN_Z_OFFSET_M = 0.2
SUPPORTED_GROUP_FORMATIONS = {'abreast', 'follow'}
DEFAULT_GROUP_SPAWN_INTERVAL = 0.15
DEFAULT_ABREAST_SPACING = 0.6
DEFAULT_FOLLOW_SPACING = 0.85


class PedSpawnManager:
    """
    Class managing all pedestrian spawners.
    """

    def __init__(self, scenario_config, carla_sim, ped_sim, spawn_event_listener=None):
        self.scenario_config = scenario_config
        self.carla_sim = carla_sim
        self.ped_sim = ped_sim
        self.spawn_event_listener = spawn_event_listener

        self.map_config = scenario_config.get('map', {})
        self.spectator_focus = scenario_config.get('walker', {}).get('spectator_focus')
        self.ped_seed = scenario_config.get('walker', {}).get('pedestrian_seed', 2000)
        self.default_ped_speed = scenario_config.get('walker', {}).get('default_speed', 1.2)
        self.variate_speed = scenario_config.get('walker', {}).get('variate_speed', 0.0)
        self.step_length = float(scenario_config.get('step_length', 0.05))
        self.ego_spawn_retry_ticks = max(
            1,
            int(scenario_config.get('walker', {}).get('ego_spawn_retry_ticks', 5)),
        )
        self.ego_spawn_retry_interval = float(
            scenario_config.get('walker', {}).get('ego_spawn_retry_interval', self.step_length)
        )
        self.ego_blueprint_age_filter = self._normalize_age_filter(
            scenario_config.get('walker', {}).get('ego_blueprint_age_filter', 'adult')
        )
        self.ego_walker_blueprints = self._walker_blueprints_matching_age(self.ego_blueprint_age_filter)
        if self.ego_blueprint_age_filter is not None and not self.ego_walker_blueprints:
            logging.warning(
                'No walker blueprints matched ego_blueprint_age_filter=%s; ego pedestrians will use the full pool.',
                self.ego_blueprint_age_filter,
            )
        waypoint_distance = scenario_config.get('walker', {}).get('waypoint_distance', 10)
        jaywalking_weight = scenario_config.get('walker', {}).get('jaywalking_weight', 2)
        amount_random_peds = scenario_config.get('walker', {}).get('random_pedestrians', 0)
        self.route_rng = random.Random(self.ped_seed)
        self.waypoint_distance = waypoint_distance
        self.jaywalking_weight = jaywalking_weight
        self.amount_random_peds = amount_random_peds
        self.path_planner = None

        # set pedestrian seed
        self.carla_sim.world.set_pedestrians_seed(self.ped_seed)

        self.scene_catalog = self._load_scene_catalog_if_needed()
        if self._needs_runtime_path_planner():
            self._ensure_path_planner()
        else:
            logging.info('Skipping runtime PedPathPlanner initialization; scene-route data is sufficient for this case.')
        self.ped_index = 0
        self.group_index = 0

        # get pedestrian spawners from scenario config
        self.ped_spawners = self._extract_ped_info()

        for i in range(amount_random_peds):
            random_ped_spawner = self._generate_random_ped_spawner()
            self.ped_spawners.append(random_ped_spawner)

        self.ego_candidate_spawners = [ped_spawner for ped_spawner in self.ped_spawners if ped_spawner.ego_candidate]
        self.ego_spawn_failure = None
        self.walker_dict = {}
        self.walker_meta_dict = {}
        self.waypoint_dict = {}
        self.scene_route_dict = {}

    def tick(self, sim_time):
        """
        Check if pedestrian spawners are ready to spawn in this simulation step and let them spawn pedestrians
        if that's the case.
        :param sim_time:
        :return:
        """
        # filter out pedestrian spawner that don't have any pedestrians left to spawn
        self.ped_spawners[:] = [ped_spawner for ped_spawner in self.ped_spawners if ped_spawner.quantity > 0]

        for ped_spawner in self.ped_spawners:
            if ped_spawner.ready_to_spawn(sim_time):
                spawn_succeeded = self._spawn_pedestrian(ped_spawner)
                if spawn_succeeded:
                    ped_spawner.record_spawn_success(sim_time)
                    continue

                will_retry = ped_spawner.record_spawn_failure(sim_time)
                if ped_spawner.ego_candidate:
                    if will_retry:
                        logging.warning(
                            'Failed to spawn ego pedestrian candidate spawner_index=%s route_id=%s '
                            'attempt=%d/%d; retrying in %.3fs.',
                            ped_spawner.spawner_index,
                            ped_spawner.route_id,
                            ped_spawner.spawn_attempts,
                            ped_spawner.max_spawn_attempts,
                            ped_spawner.retry_interval,
                        )
                    elif self.ego_spawn_failure is None:
                        self.ego_spawn_failure = {
                            'spawner_index': int(ped_spawner.spawner_index),
                            'spawn_member_index': int(ped_spawner.spawn_member_index),
                            'route_id': ped_spawner.route_id,
                            'behavior_name': ped_spawner.behavior_name,
                            'attempts': int(ped_spawner.spawn_attempts),
                        }
                        logging.error(
                            'Failed to spawn ego pedestrian candidate spawner_index=%s route_id=%s '
                            'after %d attempts.',
                            ped_spawner.spawner_index,
                            ped_spawner.route_id,
                            ped_spawner.spawn_attempts,
                        )

    def unresolved_ego_abort_message(self, sequence_manager=None):
        if not self.ego_candidate_spawners:
            return None

        if sequence_manager is not None and sequence_manager.resolved_ego is not None:
            return None

        if any(ped_spawner.quantity > 0 for ped_spawner in self.ego_candidate_spawners):
            return None

        if self.ego_spawn_failure is not None:
            failure = self.ego_spawn_failure
            return (
                'Aborting scenario because ego pedestrian could not be spawned after '
                f"{failure['attempts']} attempts "
                f"(spawner_index={failure['spawner_index']}, "
                f"spawn_member_index={failure['spawn_member_index']}, "
                f"route_id={failure['route_id']}, behavior={failure['behavior_name']})."
            )

        return (
            'Aborting scenario because all ego candidate spawners were consumed but the declared ego '
            'was never resolved.'
        )

    def _extract_ped_info(self):
        """
        Extracts pedestrian spawner information from the configuration file and converts coordinates if necessary.
        """

        ped_spawner_config = self.scenario_config.get('walker', {}).get('ped_spawner')

        ped_spawners = []
        if ped_spawner_config is not None:
            for spawner_index, spawn_point in enumerate(ped_spawner_config):
                if self._spawn_point_uses_scene_route(spawn_point):
                    ped_spawners.extend(self._generate_scene_ped_spawners(spawn_point, spawner_index))
                    continue

                spawn_location = np.array(spawn_point['spawn_location'], dtype=float)
                speed = spawn_point.get('speed', self.default_ped_speed)
                destination = np.array(spawn_point['destination'])

                # generate route or use manually defined waypoints depending on configuration
                generate_route = spawn_point.get('generate_route')
                if generate_route:
                    waypoint_tuples = self._ensure_path_planner().generate_route(spawn_location, destination,
                                                                       GraphType[generate_route])
                    waypoints = np.array([w[0] for w in waypoint_tuples])
                    crossing_road_bools = [w[1] for w in waypoint_tuples]
                else:
                    wp_list = spawn_point.get('waypoints', [])
                    wp_list.append(destination)
                    waypoints = np.array(wp_list)

                    crossing_road_bools = spawn_point.get('crossing_road_bools', [False] * len(waypoints))
                    if len(waypoints) != len(crossing_road_bools):
                        logging.warning('Length of waypoints and crossing_road_bools is not equal! '
                                        'Waypoints may get cut of!')

                blueprint = spawn_point.get('blueprint')
                quantity = spawn_point.get('quantity', 1)
                spawn_time = spawn_point.get('spawn_time', 0.0)
                spawn_interval = spawn_point.get('spawn_interval', 3.0)
                crossing_speed_factor = spawn_point.get('crossing_speed_factor', 1.5)
                crossing_safety_margin = spawn_point.get('crossing_safety_margin', 1.5)
                ego_candidate = bool(spawn_point.get('ego', False))
                ego_member_index = int(spawn_point.get('ego_member_index', 0))

                for ped_index in range(quantity):
                    is_ego_candidate = ego_candidate and ped_index == ego_member_index
                    ped_spawner = self._build_ped_spawner(
                        spawn_location,
                        waypoints,
                        crossing_road_bools,
                        speed,
                        blueprint,
                        1,
                        spawn_time + ped_index * spawn_interval,
                        spawn_interval,
                        crossing_speed_factor,
                        crossing_safety_margin,
                        spawner_index=spawner_index,
                        spawn_member_index=ped_index,
                        ego_candidate=is_ego_candidate,
                        max_spawn_attempts=self.ego_spawn_retry_ticks if is_ego_candidate else 1,
                        retry_interval=self.ego_spawn_retry_interval if is_ego_candidate else spawn_interval,
                    )
                    ped_spawners.append(ped_spawner)

        return ped_spawners

    def _load_scene_catalog_if_needed(self):
        ped_spawner_config = self.scenario_config.get('walker', {}).get('ped_spawner') or []
        if not self.map_config.get('scene_id') and not any(spawn_point.get('scene_id') for spawn_point in ped_spawner_config):
            if not any(
                spawn_point.get('behavior') or spawn_point.get('route_type') or spawn_point.get('route_id')
                for spawn_point in ped_spawner_config
            ):
                return None

        try:
            return SceneCatalog.from_map_config(self.map_config)
        except SceneCatalogError as exc:
            raise ValueError(str(exc)) from exc

    def _needs_runtime_path_planner(self):
        if self.amount_random_peds > 0:
            return True

        ped_spawner_config = self.scenario_config.get('walker', {}).get('ped_spawner') or []
        for spawn_point in ped_spawner_config:
            if self._spawn_point_uses_scene_route(spawn_point):
                continue
            if spawn_point.get('generate_route'):
                return True

        return False

    def _ensure_path_planner(self):
        if self.path_planner is None:
            logging.info(
                'Initializing runtime PedPathPlanner waypoint_distance=%s jaywalking_weight=%s.',
                self.waypoint_distance,
                self.jaywalking_weight,
            )
            self.path_planner = PedPathPlanner(
                self.carla_sim.carla_map,
                self.waypoint_distance,
                self.jaywalking_weight,
            )
        return self.path_planner

    @staticmethod
    def _spawn_point_uses_scene_route(spawn_point):
        has_route_filter = spawn_point.get('route_id') is not None or spawn_point.get('route_type') is not None
        has_behavior = spawn_point.get('behavior') is not None
        formation = spawn_point.get('formation')
        has_manual_route = (
            spawn_point.get('spawn_location') is not None
            or spawn_point.get('destination') is not None
            or spawn_point.get('generate_route') is not None
            or spawn_point.get('waypoints') is not None
        )
        if formation is not None and not has_behavior:
            raise ValueError('walker.ped_spawner.formation requires scene-based route selection with '
                             'walker.ped_spawner.behavior.')
        if has_route_filter and not has_behavior:
            raise ValueError('Scene-based route selection requires walker.ped_spawner.behavior when route_type '
                             'or route_id is configured.')
        if has_behavior and has_manual_route:
            raise ValueError('walker.ped_spawner.behavior cannot be combined with manual spawn_location, '
                             'destination, generate_route or waypoints.')
        return has_behavior

    @staticmethod
    def _validate_group_spawn_point(spawn_point):
        formation = spawn_point.get('formation')
        if formation is None:
            return None

        if formation not in SUPPORTED_GROUP_FORMATIONS:
            raise ValueError('walker.ped_spawner.formation must be one of '
                             f'{sorted(SUPPORTED_GROUP_FORMATIONS)}.')

        quantity = spawn_point.get('quantity', 1)
        if quantity not in (2, 3):
            raise ValueError('walker.ped_spawner.formation only supports quantity=2 or quantity=3.')

        route_id = spawn_point.get('route_id')
        if route_id is None:
            raise ValueError('walker.ped_spawner.formation requires walker.ped_spawner.route_id so all group '
                             'members share the same scene route.')

        return formation

    @staticmethod
    def _formation_spacing(spawn_point, formation):
        spacing = spawn_point.get('formation_spacing')
        if spacing is not None:
            spacing = float(spacing)
        elif formation == 'abreast':
            spacing = DEFAULT_ABREAST_SPACING
        else:
            spacing = DEFAULT_FOLLOW_SPACING

        if spacing <= 0.0:
            raise ValueError('walker.ped_spawner.formation_spacing must be > 0.')

        return spacing

    @staticmethod
    def _formation_offsets(formation, quantity, spacing):
        if formation == 'abreast':
            if quantity == 2:
                lateral_offsets = (-0.5 * spacing, 0.5 * spacing)
            else:
                lateral_offsets = (-spacing, 0.0, spacing)
            return [{'lateral_offset': offset, 'longitudinal_offset': 0.0} for offset in lateral_offsets]

        return [
            {'lateral_offset': 0.0, 'longitudinal_offset': spacing * index}
            for index in range(quantity)
        ]

    @staticmethod
    def _polyline_lateral_offsets(polyline_points, lateral_offset):
        points = np.array(polyline_points, dtype=float, copy=True)
        if len(points) < 2 or abs(lateral_offset) <= 1e-6:
            return points

        tangent_vectors = np.zeros((len(points), 2), dtype=float)
        for index in range(len(points)):
            if index == 0:
                delta = points[1, :2] - points[0, :2]
            elif index == len(points) - 1:
                delta = points[-1, :2] - points[-2, :2]
            else:
                delta = points[index + 1, :2] - points[index - 1, :2]

            delta_norm = float(np.linalg.norm(delta))
            if delta_norm <= 1e-6:
                tangent_vectors[index] = np.array([1.0, 0.0], dtype=float)
            else:
                tangent_vectors[index] = delta / delta_norm

        lateral_vectors = np.column_stack((-tangent_vectors[:, 1], tangent_vectors[:, 0]))
        points[:, :2] += lateral_vectors * lateral_offset
        return points

    def _build_group_member_route(self, route, crossing_road_bools, slot_offset):
        route_waypoints = route.get('path_waypoints')
        if route_waypoints:
            polyline_points = np.array(route_waypoints, dtype=float, copy=True)
        else:
            polyline_points = np.array(
                [route['spawn_location']] + route['config_waypoints'] + [route['destination']],
                dtype=float,
            )

        lateral_offset = slot_offset.get('lateral_offset', 0.0)
        if abs(lateral_offset) > 1e-6:
            polyline_points = self._polyline_lateral_offsets(polyline_points, lateral_offset)

        spawn_transform = route.get('spawn_transform') or {}
        spawn_location = self._project_route_spawn_location(
            polyline_points[0],
            polyline_points[1],
            spawn_transform,
        )
        polyline_points[0] = spawn_location
        longitudinal_offset = slot_offset.get('longitudinal_offset', 0.0)
        if abs(longitudinal_offset) > 1e-6:
            forward, _ = self._spawn_frame(spawn_location, polyline_points[1])
            spawn_location[:2] -= forward * longitudinal_offset

        waypoints = np.array(polyline_points[1:], dtype=float, copy=True)
        scene_route = SceneRoute(polyline_points, crossing_road_bools) if route_waypoints else None
        return spawn_location, waypoints, scene_route

    def _project_route_spawn_location(self, spawn_location, first_waypoint, spawn_transform=None):
        if not spawn_transform or not spawn_transform.get('requires_runtime_projection', False):
            return np.array(spawn_location, dtype=float)

        location = carla.Location(
            float(spawn_location[0]),
            float(spawn_location[1]),
            float(spawn_location[2]) if len(spawn_location) > 2 else 0.0,
        )
        for lane_type in (carla.LaneType.Sidewalk, carla.LaneType.Shoulder, carla.LaneType.Any):
            try:
                waypoint = self.carla_sim.carla_map.get_waypoint(
                    location,
                    project_to_road=True,
                    lane_type=lane_type,
                )
            except RuntimeError:
                waypoint = None
            if waypoint is None:
                continue

            snapped = waypoint.transform.location
            projected = np.array(
                [
                    float(snapped.x),
                    float(snapped.y),
                    float(snapped.z) + ROUTE_SPAWN_Z_OFFSET_M,
                ],
                dtype=float,
            )
            logging.info(
                'Projected reverse route spawn from [%.3f, %.3f, %.3f] to [%.3f, %.3f, %.3f].',
                float(spawn_location[0]),
                float(spawn_location[1]),
                float(spawn_location[2]) if len(spawn_location) > 2 else 0.0,
                projected[0],
                projected[1],
                projected[2],
            )
            return projected

        fallback = np.array(spawn_location, dtype=float, copy=True)
        if fallback.shape[0] >= 3:
            fallback[2] += ROUTE_SPAWN_Z_OFFSET_M
        logging.warning(
            'Failed to project reverse route spawn [%.3f, %.3f, %.3f]; using raw point with z offset.',
            float(spawn_location[0]),
            float(spawn_location[1]),
            float(spawn_location[2]) if len(spawn_location) > 2 else 0.0,
        )
        return fallback

    def _extract_scene_id(self, spawn_point):
        scene_id = spawn_point.get('scene_id', self.map_config.get('scene_id'))
        if scene_id is None:
            raise ValueError('Scene-based route selection requires map.scene_id or walker.ped_spawner.scene_id.')
        return scene_id

    def _build_ped_spawner(
        self,
        spawn_location,
        waypoints,
        crossing_road_bools,
        speed,
        blueprint,
        quantity,
        spawn_time,
        spawn_interval,
        crossing_speed_factor,
        crossing_safety_margin,
        scene_route=None,
        group_id=-1,
        group_member_index=0,
        spawner_index=-1,
        spawn_member_index=0,
        route_id=None,
        behavior_name=None,
        ego_candidate=False,
        max_spawn_attempts=1,
        retry_interval=None,
    ):
        return PedSpawner(
            spawn_location,
            waypoints,
            crossing_road_bools,
            speed,
            blueprint,
            quantity,
            spawn_time,
            spawn_interval,
            crossing_speed_factor,
            crossing_safety_margin,
            scene_route=scene_route,
            group_id=group_id,
            group_member_index=group_member_index,
            spawner_index=spawner_index,
            spawn_member_index=spawn_member_index,
            route_id=route_id,
            behavior_name=behavior_name,
            ego_candidate=ego_candidate,
            max_spawn_attempts=max_spawn_attempts,
            retry_interval=retry_interval,
        )

    def _scene_route_crossing_bools(self, scene_id, route):
        crossing_road_bools = list(route.get('config_crossing_road_bools', []))
        edge_ids = route.get('edge_ids', [])
        if len(crossing_road_bools) != len(edge_ids):
            return crossing_road_bools

        if not route.get('attributes', {}).get('crossing_required', False):
            return crossing_road_bools

        scene_edges = {
            edge['edge_id']: edge
            for edge in self.scene_catalog.get_scene(scene_id).get('edges', [])
        }
        adjusted_bools = []
        for crossing_road, edge_id in zip(crossing_road_bools, edge_ids):
            edge = scene_edges.get(edge_id, {})
            is_shoulder_transition = edge.get('type') == 'SIDEWALK' and (
                edge.get('entry_lane_type') == 'Shoulder' or edge.get('exit_lane_type') == 'Shoulder'
            )
            adjusted_bools.append(bool(crossing_road or is_shoulder_transition))

        return adjusted_bools

    def _generate_scene_ped_spawners(self, spawn_point, spawner_index):
        if self.scene_catalog is None:
            raise ValueError('Scene-based route selection requires a valid scene catalog.')

        scene_id = self._extract_scene_id(spawn_point)
        behavior_name = spawn_point['behavior']
        formation = self._validate_group_spawn_point(spawn_point)
        behavior_profile = self.scene_catalog.get_behavior_profile(behavior_name)
        quantity = spawn_point.get('quantity', 1)
        spawn_time = spawn_point.get('spawn_time', 0.0)
        default_spawn_interval = DEFAULT_GROUP_SPAWN_INTERVAL if formation is not None else 3.0
        spawn_interval = spawn_point.get('spawn_interval', default_spawn_interval)
        route_type = spawn_point.get('route_type')
        route_id = spawn_point.get('route_id')
        blueprint = spawn_point.get('blueprint')

        if 'speed' in spawn_point:
            speed = spawn_point['speed']
        else:
            base_speed = spawn_point.get('base_speed', self.default_ped_speed)
            speed = base_speed * behavior_profile.get('speed_multiplier', 1.0)

        crossing_speed_factor = spawn_point.get(
            'crossing_speed_factor',
            behavior_profile.get('crossing_speed_factor', 1.5),
        )
        crossing_safety_margin = spawn_point.get(
            'crossing_safety_margin',
            behavior_profile.get('crossing_safety_margin', 1.5),
        )

        candidate_routes = self.scene_catalog.get_routes(
            scene_id=scene_id,
            behavior_name=behavior_name,
            route_type=route_type,
            route_id=route_id,
        )
        ped_spawners = []
        if formation is not None:
            group_id = self.group_index
            self.group_index += 1
            route = self.route_rng.choice(candidate_routes)
            crossing_road_bools = self._scene_route_crossing_bools(scene_id, route)
            spacing = self._formation_spacing(spawn_point, formation)
            formation_offsets = self._formation_offsets(formation, quantity, spacing)

            for ped_index, slot_offset in enumerate(formation_offsets):
                is_ego_candidate = bool(spawn_point.get('ego', False)) and ped_index == int(
                    spawn_point.get('ego_member_index', 0)
                )
                member_spawn_location, member_waypoints, scene_route = self._build_group_member_route(
                    route,
                    crossing_road_bools,
                    slot_offset,
                )
                ped_spawners.append(
                    self._build_ped_spawner(
                        member_spawn_location,
                        member_waypoints,
                        crossing_road_bools,
                        speed,
                        blueprint,
                        1,
                        spawn_time + ped_index * spawn_interval,
                        spawn_interval,
                        crossing_speed_factor,
                        crossing_safety_margin,
                        scene_route=scene_route,
                        group_id=group_id,
                        group_member_index=ped_index,
                        spawner_index=spawner_index,
                        spawn_member_index=ped_index,
                        route_id=route['route_id'],
                        behavior_name=behavior_name,
                        ego_candidate=is_ego_candidate,
                        max_spawn_attempts=self.ego_spawn_retry_ticks if is_ego_candidate else 1,
                        retry_interval=self.ego_spawn_retry_interval if is_ego_candidate else spawn_interval,
                    )
                )
            logging.info(
                'Selected group scene route %s for scene=%s behavior=%s formation=%s quantity=%d.',
                route['route_id'],
                scene_id,
                behavior_name,
                formation,
                quantity,
            )
            return ped_spawners

        for ped_index in range(quantity):
            is_ego_candidate = bool(spawn_point.get('ego', False)) and ped_index == int(
                spawn_point.get('ego_member_index', 0)
            )
            route = self.route_rng.choice(candidate_routes)
            crossing_road_bools = self._scene_route_crossing_bools(scene_id, route)
            route_waypoints = route.get('path_waypoints')
            if route_waypoints:
                waypoints = np.array(route_waypoints[1:], dtype=float)
                scene_route = SceneRoute(route_waypoints, crossing_road_bools)
            else:
                waypoints = np.array(route['config_waypoints'] + [route['destination']], dtype=float)
                scene_route = None
            spawn_transform = route.get('spawn_transform') or {}
            spawn_location = self._project_route_spawn_location(
                spawn_transform.get('location', route['spawn_location']),
                waypoints[0],
                spawn_transform,
            )
            ped_spawners.append(
                self._build_ped_spawner(
                    spawn_location,
                    waypoints,
                    crossing_road_bools,
                    speed,
                    blueprint,
                    1,
                    spawn_time + ped_index * spawn_interval,
                    spawn_interval,
                    crossing_speed_factor,
                    crossing_safety_margin,
                    scene_route=scene_route,
                    spawner_index=spawner_index,
                    spawn_member_index=ped_index,
                    route_id=route['route_id'],
                    behavior_name=behavior_name,
                    ego_candidate=is_ego_candidate,
                    max_spawn_attempts=self.ego_spawn_retry_ticks if is_ego_candidate else 1,
                    retry_interval=self.ego_spawn_retry_interval if is_ego_candidate else spawn_interval,
                )
            )
            logging.info(
                'Selected scene route %s for scene=%s behavior=%s.',
                route['route_id'],
                scene_id,
                behavior_name,
            )

        return ped_spawners

    @staticmethod
    def _spawn_frame(spawn_location, first_waypoint):
        direction = np.array(first_waypoint[:2], dtype=float) - np.array(spawn_location[:2], dtype=float)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            forward = np.array([1.0, 0.0], dtype=float)
        else:
            forward = direction / direction_norm
        lateral = np.array([-forward[1], forward[0]], dtype=float)
        return forward, lateral

    def _spawn_location_candidates(self, ped_spawner):
        base_location = np.array(ped_spawner.spawn_location, dtype=float, copy=True)
        forward, lateral = self._spawn_frame(base_location, ped_spawner.first_waypoint)

        seen = set()
        for z_lift in AUTO_SPAWN_Z_LIFTS:
            for longitudinal_offset, lateral_offset in AUTO_SPAWN_OFFSET_PAIRS:
                candidate = np.array(base_location, dtype=float, copy=True)
                candidate[:2] += forward * longitudinal_offset + lateral * lateral_offset
                candidate[2] += z_lift
                key = tuple(np.round(candidate, 4))
                if key in seen:
                    continue
                seen.add(key)
                yield candidate

    def _spawn_actor_with_fallbacks(self, walker_bp, ped_spawner):
        base_location = np.array(ped_spawner.spawn_location, dtype=float, copy=True)

        for candidate_location in self._spawn_location_candidates(ped_spawner):
            candidate_transform = ped_spawner.generate_carla_spawn_point(candidate_location)
            actor = self.carla_sim.world.try_spawn_actor(walker_bp, candidate_transform)
            if actor is None:
                continue

            if not np.allclose(candidate_location, base_location):
                delta = candidate_location - base_location
                logging.info(
                    'Adjusted spawn for %s by dx=%.3f dy=%.3f dz=%.3f.',
                    ped_spawner.blueprint or 'random_walker',
                    delta[0],
                    delta[1],
                    delta[2],
                )
            ped_spawner.set_spawn_location(candidate_location)
            return actor.id

        return -1

    def _generate_random_ped_spawner(self, speed=1.0, blueprint=None, quantity=1, spawn_time=0.0, spawn_interval=1.0,
                                     crossing_speed_factor=1.5, crossing_safety_margin=1.5,
                                     graph_type=GraphType.JAYWALKING_AT_JUNCTION):
        """
        Generate pedestrian spawner with random spawn position and random destination
        """

        spawn_point = self.carla_sim.world.get_random_location_from_navigation()
        destination = self.carla_sim.world.get_random_location_from_navigation()

        waypoint_tuples = self._ensure_path_planner().generate_route(spawn_point, destination, graph_type, with_origin=True)
        np_spawn_loc = waypoint_tuples.pop(0)[0]
        waypoints = np.array([w[0] for w in waypoint_tuples])
        crossing_road_bools = [w[1] for w in waypoint_tuples]

        ped_spawner = PedSpawner(np_spawn_loc, waypoints, crossing_road_bools, speed, blueprint, quantity,
                                 spawn_time, spawn_interval, crossing_speed_factor, crossing_safety_margin)

        return ped_spawner

    @staticmethod
    def _normalize_age_filter(age_filter):
        if age_filter is None:
            return None
        normalized = str(age_filter).strip().lower()
        if normalized in {'', 'any', 'all', 'none'}:
            return None
        return normalized

    @staticmethod
    def _blueprint_attribute_value(blueprint, attr_name):
        try:
            if hasattr(blueprint, 'has_attribute') and blueprint.has_attribute(attr_name):
                attr = blueprint.get_attribute(attr_name)
                return attr.as_str() if hasattr(attr, 'as_str') else str(attr)
        except Exception:
            return None

        try:
            for attr in blueprint:
                if attr.id == attr_name:
                    return attr.as_str() if hasattr(attr, 'as_str') else str(attr)
        except Exception:
            return None

        return None

    def _walker_blueprints_matching_age(self, age_filter):
        if age_filter is None:
            return []

        matches = []
        for blueprint in self.carla_sim.walker_blueprints:
            age = self._blueprint_attribute_value(blueprint, 'age')
            if age is not None and str(age).strip().lower() == age_filter:
                matches.append(blueprint)
        return matches

    def _select_walker_blueprint(self, ped_spawner):
        if ped_spawner.blueprint:
            return self.carla_sim.walker_blueprints.find(ped_spawner.blueprint)

        if ped_spawner.ego_candidate and self.ego_walker_blueprints:
            return random.choice(self.ego_walker_blueprints)

        return random.choice(self.carla_sim.walker_blueprints)

    def _spawn_pedestrian(self, ped_spawner):
        """
        Spawn pedestrian in both Carla and pedestrian simulation.
        """
        ped_name = self._generate_ped_name()

        random.seed(self.ped_seed)
        walker_bp = self._select_walker_blueprint(ped_spawner)

        if walker_bp.has_attribute('role_name'):
            walker_bp.set_attribute('role_name', ped_name)

        # Spawn pedestrian in CARLA. Route start nodes are not guaranteed to be valid actor spawn points,
        # so search a small local neighborhood before giving up.
        actor_id = self._spawn_actor_with_fallbacks(walker_bp, ped_spawner)

        if self.variate_speed != 0.0:
            ped_spawner.target_speed += random.uniform(-self.variate_speed, self.variate_speed)

        # increment ped seed so next walker has a different random blueprint and speed variation
        self.ped_seed += 1

        if actor_id == -1:
            logging.info(f'Failed to spawn pedestrian {ped_name}.')
            return False
        else:
            ped_radius = self.carla_sim.get_ped_radius(actor_id)
            if ped_spawner.scene_route is not None:
                scene_route_follower = SceneRouteFollower(ped_spawner.scene_route, ped_spawner.spawn_location)
                initial_target = scene_route_follower.update(ped_spawner.spawn_location, ped_spawner.target_speed)
                initial_ped_state = ped_spawner.generate_scene_route_ped_state(
                    ped_name,
                    actor_id,
                    ped_radius,
                    initial_target,
                )
                remaining_waypoints = []
                self.scene_route_dict[ped_name] = scene_route_follower
            else:
                initial_ped_state, remaining_waypoints = ped_spawner.generate_ped_state(ped_name, actor_id, ped_radius)

            # spawn pedestrian in pedestrian simulation
            self.ped_sim.spawn_pedestrian(initial_ped_state)

            self.walker_dict[ped_name] = actor_id
            spawn_metadata = {
                'spawner_index': int(ped_spawner.spawner_index),
                'spawn_member_index': int(ped_spawner.spawn_member_index),
                'group_id': int(ped_spawner.group_id),
                'group_member_index': int(ped_spawner.group_member_index),
                'route_id': ped_spawner.route_id,
                'behavior_name': ped_spawner.behavior_name,
                'ego_candidate': bool(ped_spawner.ego_candidate),
            }
            self.walker_meta_dict[ped_name] = spawn_metadata
            # it's important to make a shallow copy of the remaining waypoints with [:], because otherwise waypoints
            # get removed for all peds of one spawner when the first pedestrian reaches the waypoint
            self.waypoint_dict[ped_name] = remaining_waypoints[:]

            if self.spawn_event_listener is not None:
                self.spawn_event_listener.on_ped_spawned(ped_name, actor_id, spawn_metadata)

            # place spectator camera behind selected pedestrian
            if self.spectator_focus == ped_name:
                spectator = self.carla_sim.world.get_spectator()
                spectator_transform = carla.Transform()
                spectator_transform.location = ped_spawner.carla_spawn_point.transform(carla.Vector3D(-2.0, 0.0, 2.0))
                spectator_transform.rotation = ped_spawner.carla_spawn_point.rotation
                spectator.set_transform(spectator_transform)

            logging.info(f'Spawned pedestrian {ped_name}.')
            return True

    def _generate_ped_name(self):
        """
        Generate pedestrian name with incrementing index.
        """
        name = 'ped_' + str(self.ped_index)
        self.ped_index += 1

        return name


class PedSpawner:
    """
    Class containing all the information necessary to spawn one or multiple pedestrians from one spawn point.
    """

    def __init__(self, spawn_location, waypoints, crossing_road_bools, speed, blueprint, quantity, spawn_time,
                 spawn_interval, crossing_speed_factor, crossing_safety_margin, scene_route=None,
                 group_id=-1, group_member_index=0, spawner_index=-1, spawn_member_index=0,
                 route_id=None, behavior_name=None, ego_candidate=False,
                 max_spawn_attempts=1, retry_interval=None):
        self.target_speed = speed
        self.blueprint = blueprint
        self.quantity = quantity
        self.spawn_interval = spawn_interval
        self.next_spawn_time = spawn_time
        self.crossing_speed_factor = crossing_speed_factor
        self.crossing_safety_margin = crossing_safety_margin
        self.scene_route = scene_route
        self.group_id = int(group_id)
        self.group_member_index = int(group_member_index)
        self.spawner_index = int(spawner_index)
        self.spawn_member_index = int(spawn_member_index)
        self.route_id = route_id
        self.behavior_name = behavior_name
        self.ego_candidate = bool(ego_candidate)
        self.max_spawn_attempts = max(1, int(max_spawn_attempts))
        self.retry_interval = float(spawn_interval if retry_interval is None else retry_interval)
        self.spawn_attempts = 0
        self.spawn_succeeded = False
        self.spawn_exhausted = False

        if crossing_road_bools[0]:
            self.initial_mode = PedMode.CROSSING_ROAD
        else:
            self.initial_mode = PedMode.WALKING_SIDEWALK

        if waypoints.ndim > 1:
            self.first_waypoint = waypoints[0]
            self.remaining_waypoint_tuples = list(zip(waypoints[1:].tolist(), crossing_road_bools[1:]))
        else:
            self.first_waypoint = waypoints
            self.remaining_waypoint_tuples = []

        self.set_spawn_location(spawn_location)

    def ready_to_spawn(self, sim_time):
        """
        Check if pedestrian spawner is ready to spawn in this simulation time step.
        :param sim_time:
        :return: True if pedestrian spawner is ready to spawn and False if not
        """
        return self.next_spawn_time <= sim_time

    def record_spawn_success(self, sim_time):
        self.spawn_succeeded = True
        self.spawn_attempts = 0
        self.quantity = max(0, self.quantity - 1)
        self.next_spawn_time = max(self.next_spawn_time, sim_time) + self.spawn_interval

    def record_spawn_failure(self, sim_time):
        self.spawn_attempts += 1
        if self.spawn_attempts >= self.max_spawn_attempts:
            self.spawn_exhausted = True
            self.quantity = max(0, self.quantity - 1)
            return False

        self.next_spawn_time = sim_time + max(self.retry_interval, 0.0)
        return True

    def generate_ped_state(self, name, carla_id, radius):
        """
        Generate pedestrian state from spawn info.
        :param name:
        :param carla_id:
        :param radius:
        :return: initial pedestrian state and remaining waypoints
        """
        ped_mode = PedModeManager(name, self.target_speed, self.initial_mode, self.crossing_speed_factor,
                                  self.crossing_safety_margin)
        ped_state = (name, carla_id, self.spawn_location, self.velocity, self.first_waypoint, ped_mode,
                     radius, self.target_speed, self.group_id, self.group_member_index)

        return ped_state, self.remaining_waypoint_tuples

    def generate_scene_route_ped_state(self, name, carla_id, radius, navigation_target):
        initial_mode = PedMode.CROSSING_ROAD if navigation_target.crossing_road else PedMode.WALKING_SIDEWALK
        ped_mode = PedModeManager(
            name,
            self.target_speed,
            initial_mode,
            self.crossing_speed_factor,
            self.crossing_safety_margin,
        )
        return (
            name,
            carla_id,
            self.spawn_location,
            self.velocity,
            np.array(navigation_target.next_waypoint, dtype=float),
            ped_mode,
            radius,
            self.target_speed,
            self.group_id,
            self.group_member_index,
        )

    def set_spawn_location(self, spawn_location):
        self.spawn_location = np.array(spawn_location, dtype=float, copy=True)
        self.carla_spawn_point = self.generate_carla_spawn_point(self.spawn_location)
        direction = self.carla_spawn_point.get_forward_vector()
        self.velocity = np.array([direction.x, direction.y, direction.z]) * self.target_speed

    def generate_carla_spawn_point(self, spawn_location=None):
        """
        Generate spawn point for Carla simulator.
        :return: Carla spawn point
        """
        if spawn_location is None:
            spawn_location = self.spawn_location
        carla_spawn_point = carla.Transform()
        carla_spawn_point.location = carla.Location(spawn_location[0],
                                                    spawn_location[1],
                                                    spawn_location[2])
        # calculate spawn rotation
        direction = self.first_waypoint - spawn_location
        ref_axis = np.array([1.0, 0.0, 0.0])
        rotation_angle = stateutils.angle_diff_2d(direction, ref_axis)
        carla_spawn_point.rotation = carla.Rotation(0, np.degrees(rotation_angle), 0)

        return carla_spawn_point

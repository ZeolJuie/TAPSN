import glob
import hashlib
import itertools
import logging
from dataclasses import dataclass
import os
import tempfile
import time
from typing import Optional

import carla
import numpy as np


def extract_sidewalk(carla_map, scenario_config):
    """
    Extracts sidewalk borders form Carla map (based on the underlying OpenDRIVE map) as a list of points.
    A cache system is used, so if the OpenDrive content of a Carla town and the obstacle resolution has not changed,
    it will read and use the stored sidewalk borders that were extracted in a previous simulation run.
    :param carla_map:
    :param scenario_config:
    :return: numpy_borders (list of numpy arrays), carla_borders (list of Carla Vector3d)
    """

    logging.info('Start extracting sidewalks.')
    start = time.time()
    # distance between extracted border points
    resolution = scenario_config.get('obstacles', {}).get('resolution', 0.1)

    # Load OpenDrive content
    opendrive_content = carla_map.to_opendrive()

    # Get hash based on content
    hash_func = hashlib.sha1()
    hash_func.update(opendrive_content.encode('UTF-8'))
    opendrive_hash = str(hash_func.hexdigest())

    # Build path for saving or loading the cached rendered map
    filename = carla_map.name.split('/')[-1] + '_' + str(resolution) + '_' + opendrive_hash + '.npz'
    dirname = os.path.join('cache', 'sidewalk_borders')
    full_path = str(os.path.join(dirname, filename))

    cache_loaded = False
    if os.path.isfile(full_path):
        try:
            # Load sidewalk borders from cache file
            logging.info('Using cached sidewalk borders.')
            loaded_file = np.load(full_path, allow_pickle=True)
            numpy_borders = [np.array(border) for border in loaded_file['borders']]
            section_info = loaded_file['section_info'].tolist()
            carla_borders = [[carla.Vector3D(p[0], p[1], 0.5) for p in border] for border in numpy_borders]
            cache_loaded = True
        except (KeyError, ValueError, OSError, EOFError) as exc:
            logging.warning('Failed to load sidewalk cache %s: %s. Rebuilding cache.', full_path, exc)
            try:
                os.remove(full_path)
            except OSError:
                pass

    if not cache_loaded:
        carla_borders, section_info = extract_sidewalk_borders(carla_map, resolution)

        # convert carla Vector3d to numpy arrays
        numpy_borders = [np.array([[vec.x, vec.y] for vec in border]) for border in carla_borders]

        # If folders path does not exist, create it
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Remove files if selected town had a previous version saved
        list_filenames = glob.glob(os.path.join(dirname, carla_map.name.split('/')[-1]) + "*")
        for town_filename in list_filenames:
            os.remove(town_filename)

        # Save sidewalk borders and section info for next executions of same map.
        # These collections are ragged, so they must be persisted as object arrays.
        fd, tmp_path = tempfile.mkstemp(dir=dirname, suffix='.npz')
        os.close(fd)
        try:
            np.savez(
                tmp_path,
                borders=np.array(numpy_borders, dtype=object),
                section_info=np.array(section_info, dtype=object),
            )
            os.replace(tmp_path, full_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    end = time.time()
    logging.info('Finished extracting sidewalks. Time: ' + str(end - start))

    return numpy_borders, section_info, carla_borders


def extract_sidewalk_borders(carla_map, resolution):
    # get topology (minimal graph of OpenDRIVE map), which consists of a list of tuples of
    # waypoints (start of road, end of road)
    carla_topology = carla_map.get_topology()

    # use only the start waypoints
    topology = [x[0] for x in carla_topology]

    # the sidewalks in junctions must be extracted separately, because they are not always attached to a lane of
    # type driving

    # get all junctions and filter out duplicates
    junctions = [w.get_junction() for w in topology if w.is_junction]
    junction_ids = set()
    filtered_junctions = []
    for j in junctions:
        if j.id not in junction_ids:
            filtered_junctions.append(j)
            junction_ids.add(j.id)

    # get all start waypoints of type sidewalk that are located in a junction
    junction_waypoints = []
    for junction in filtered_junctions:
        waypoint_tuples = junction.get_waypoints(carla.LaneType.Sidewalk)
        start_waypoints = [x[0] for x in waypoint_tuples]
        junction_waypoints.extend(start_waypoints)

    # filter out all duplicates and all waypoints that are part of a junction to avoid duplicates when merging with
    # separately extracted junction waypoints
    topology = [w for w in topology if w.is_junction is False]
    waypoint_ids = set()
    filtered_waypoints = []
    for w in topology:
        if w.id not in waypoint_ids:
            filtered_waypoints.append(w)
            waypoint_ids.add(w.id)

    # merge start waypoints inside and outside of junctions
    filtered_waypoints.extend(junction_waypoints)

    carla_borders = []
    section_info = []
    for waypoint in filtered_waypoints:
        waypoints = [waypoint]

        # Generate waypoints of a road id. Stop when road id differs
        nxt = waypoint.next(resolution)
        if len(nxt) > 0:
            nxt = nxt[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(resolution)
                if len(nxt) > 0:
                    nxt = nxt[0]
                else:
                    break

        middle_wp = waypoints[len(waypoints) // 2]
        middle_loc = np.array([middle_wp.transform.location.x, middle_wp.transform.location.y])
        section_length = len(waypoints) * resolution

        sidewalk_waypoints = []
        for w in waypoints:
            if w.lane_type == carla.LaneType.Sidewalk:
                sidewalk_waypoints.append(w)

            # Check for sidewalk lane type until there are no waypoints by going left
            l = w.get_left_lane()
            while l and l.lane_type != carla.LaneType.Driving:

                if l.lane_type == carla.LaneType.Sidewalk:
                    sidewalk_waypoints.append(l)

                l = l.get_left_lane()

            # Check for sidewalk lane type until there are no waypoints by going right
            r = w.get_right_lane()
            while r and r.lane_type != carla.LaneType.Driving:

                if r.lane_type == carla.LaneType.Sidewalk:
                    sidewalk_waypoints.append(r)

                r = r.get_right_lane()

        if sidewalk_waypoints:
            border_left = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in sidewalk_waypoints]
            border_right = [lateral_shift(w.transform, w.lane_width * 0.5) for w in sidewalk_waypoints]

            carla_borders.append(border_left)
            section_info.append([middle_loc, section_length])

            carla_borders.append(border_right)
            section_info.append([middle_loc, section_length])

    return carla_borders, section_info


def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    transform.location.z = 0.5
    return transform.location + shift * transform.get_forward_vector()


def _object_bottom_z(bounding_box, transform):
    del transform
    return bounding_box.location.z - bounding_box.extent.z


def _nearest_ground_height_from_waypoint(carla_map, location):
    lane_type_order = [
        carla.LaneType.Sidewalk,
        carla.LaneType.Driving,
        carla.LaneType.Any,
    ]
    for lane_type in lane_type_order:
        try:
            waypoint = carla_map.get_waypoint(location, project_to_road=True, lane_type=lane_type)
        except RuntimeError:
            waypoint = None
        if waypoint is not None:
            return waypoint.transform.location.z, lane_type
    return None, None


@dataclass(frozen=True)
class StaticObstacleGeometrySource:
    object_id: int
    label_name: str
    bounding_box: carla.BoundingBox
    transform: Optional[carla.Transform]


def _bounding_box_reference_transform(bounding_box, transform):
    if transform is None:
        return carla.Transform(
            carla.Location(
                x=float(bounding_box.location.x),
                y=float(bounding_box.location.y),
                z=float(bounding_box.location.z),
            ),
            bounding_box.rotation,
        )

    tolerance = (bounding_box.location - transform.location) * 0.1
    object_loc = transform.location + tolerance
    rot = carla.Rotation(-bounding_box.rotation.pitch, -bounding_box.rotation.yaw, -bounding_box.rotation.roll)
    if bb_contains(bounding_box, object_loc, carla.Transform(rotation=rot)):
        return carla.Transform(
            carla.Location(
                x=float(transform.location.x),
                y=float(transform.location.y),
                z=float(bounding_box.location.z),
            ),
            transform.rotation,
        )
    return carla.Transform(
        carla.Location(
            x=float(bounding_box.location.x),
            y=float(bounding_box.location.y),
            z=float(bounding_box.location.z),
        ),
        bounding_box.rotation,
    )


def _synthetic_level_bbox_object_id(label_name, bbox_index, bounding_box):
    digest = hashlib.sha1(
        (
            f"{label_name}|{bbox_index}|"
            f"{float(bounding_box.location.x):.6f}|{float(bounding_box.location.y):.6f}|{float(bounding_box.location.z):.6f}|"
            f"{float(bounding_box.extent.x):.6f}|{float(bounding_box.extent.y):.6f}|{float(bounding_box.extent.z):.6f}|"
            f"{float(bounding_box.rotation.pitch):.6f}|{float(bounding_box.rotation.yaw):.6f}|{float(bounding_box.rotation.roll):.6f}"
        ).encode('utf-8')
    ).hexdigest()
    return -int(digest[:15], 16)


def _static_obstacle_geometry_key(label_name, bounding_box):
    return (
        label_name,
        round(float(bounding_box.location.x), 3),
        round(float(bounding_box.location.y), 3),
        round(float(bounding_box.location.z), 3),
        round(float(bounding_box.extent.x), 3),
        round(float(bounding_box.extent.y), 3),
        round(float(bounding_box.extent.z), 3),
        round(float(bounding_box.rotation.pitch), 3),
        round(float(bounding_box.rotation.yaw), 3),
        round(float(bounding_box.rotation.roll), 3),
    )


def _iter_static_obstacle_sources(carla_world, label_names):
    sources_by_key = {}
    for label_name in label_names:
        label = getattr(carla.CityObjectLabel, label_name, None)
        if label is None:
            logging.info('Obstacle extraction: skip missing label=%s', label_name)
            continue
        logging.info('Obstacle extraction: query env_objects label=%s start', label_name)
        label_objects = carla_world.get_environment_objects(label)
        logging.info('Obstacle extraction: query env_objects label=%s done count=%d', label_name, len(label_objects))
        for obj in label_objects:
            source = StaticObstacleGeometrySource(
                object_id=int(obj.id),
                label_name=label_name,
                bounding_box=obj.bounding_box,
                transform=obj.transform,
            )
            key = _static_obstacle_geometry_key(label_name, obj.bounding_box)
            sources_by_key.setdefault(key, source)
        try:
            logging.info('Obstacle extraction: query level_bbs label=%s start', label_name)
            level_bbs = list(carla_world.get_level_bbs(label))
            logging.info('Obstacle extraction: query level_bbs label=%s done count=%d', label_name, len(level_bbs))
        except RuntimeError as exc:
            logging.info('Obstacle extraction: query level_bbs label=%s failed error=%s', label_name, exc)
            level_bbs = []
        for bbox_index, bounding_box in enumerate(level_bbs):
            key = _static_obstacle_geometry_key(label_name, bounding_box)
            if key in sources_by_key:
                continue
            sources_by_key[key] = StaticObstacleGeometrySource(
                object_id=_synthetic_level_bbox_object_id(label_name, bbox_index, bounding_box),
                label_name=label_name,
                bounding_box=bounding_box,
                transform=None,
            )
    logging.info('Obstacle extraction: merged geometry sources unique_count=%d', len(sources_by_key))
    return list(sources_by_key.values())


def _build_obstacle_geometry(source, resolution, ellipse_shape):
    bb = source.bounding_box
    transform = _bounding_box_reference_transform(bb, source.transform)
    object_bottom_z = _object_bottom_z(bb, transform)
    center = np.array([transform.location.x, transform.location.y])

    if ellipse_shape:
        if source.label_name == 'Poles' and source.transform is not None:
            extent = min([bb.extent.x, bb.extent.y])
            extent_x = extent
            extent_y = extent
        else:
            extent_x = bb.extent.x
            extent_y = bb.extent.y
        border_points = generate_ellipse_border(transform, extent_x, extent_y, resolution)
        return center, border_points, object_bottom_z

    vertices = bb.get_local_vertices()[::2]
    borders = []
    border_lengths = []
    combinations = itertools.combinations(vertices, 2)
    for start, end in combinations:
        start_point = np.array([start.x, start.y])
        end_point = np.array([end.x, end.y])
        length = np.linalg.norm(end_point - start_point)
        border_lengths.append(length)
        samples = max(2, int(length / resolution))
        border_line = np.column_stack((np.linspace(start_point[0], end_point[0], samples),
                                       np.linspace(start_point[1], end_point[1], samples)))
        borders.append(border_line)
    if len(vertices) != 4:
        return None, None, object_bottom_z
    indices = np.argpartition(border_lengths, 4)[:4]
    borders = [borders[i] for i in indices]
    border_points = [transform.transform(carla.Location(float(p[0]), float(p[1]), 0.0)) for border_line in borders for p in border_line]
    return center, border_points, object_bottom_z


def extract_obstacles(carla_world, scenario_config):
    """
    Extracts obstacles from Carla world map and generates border points around its bounding box.
    Depending on the configuration, the border points are either in the shap of a rectangle or of an ellipse.
    """

    resolution = scenario_config.get('obstacles', {}).get('resolution', 0.1)
    ellipse_shape = scenario_config.get('obstacles', {}).get('ellipse_shape', True)
    max_obstacle_clearance = scenario_config.get('obstacles', {}).get('max_obstacle_z_pos', 1.0)
    carla_map = carla_world.get_map()
    logging.info(
        'Obstacle extraction: start resolution=%s ellipse_shape=%s max_obstacle_clearance=%s',
        resolution,
        ellipse_shape,
        max_obstacle_clearance,
    )

    label_names = ['Static', 'Poles', 'Walls']
    if hasattr(carla.CityObjectLabel, 'Vehicles'):
        label_names.append('Vehicles')
    else:
        label_names.extend(['Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle', 'Train'])
    logging.info('Obstacle extraction: querying labels=%s', label_names)

    geometry_sources = _iter_static_obstacle_sources(carla_world, label_names)

    carla_obstacle_borders = []
    obstacle_positions = []

    for index, source in enumerate(geometry_sources):
        if index > 0 and index % 250 == 0:
            logging.info('Obstacle extraction: processed %d/%d objects', index, len(geometry_sources))
        probe_transform = _bounding_box_reference_transform(source.bounding_box, source.transform)
        object_bottom_z = _object_bottom_z(source.bounding_box, probe_transform)
        ground_z, ground_lane_type = _nearest_ground_height_from_waypoint(carla_map, probe_transform.location)
        if ground_z is None:
            logging.info(
                'Obstacle extraction: skip object_id=%s label=%s reason=no_ground_waypoint loc=(%.3f, %.3f, %.3f)',
                source.object_id,
                source.label_name,
                probe_transform.location.x,
                probe_transform.location.y,
                probe_transform.location.z,
            )
            continue

        clearance = object_bottom_z - ground_z
        if clearance > max_obstacle_clearance:
            logging.info(
                'Obstacle extraction: skip object_id=%s label=%s clearance=%.3f bottom_z=%.3f ground_z=%.3f lane_type=%s',
                source.object_id,
                source.label_name,
                clearance,
                object_bottom_z,
                ground_z,
                ground_lane_type,
            )
            continue

        center, border_points, _ = _build_obstacle_geometry(source, resolution, ellipse_shape)
        if border_points is None:
            continue

        carla_obstacle_borders.append(border_points)
        obstacle_positions.append(center)

    numpy_obstacle_borders = [np.array([[vec.x, vec.y] for vec in border]) for border in carla_obstacle_borders]
    logging.info(
        'Obstacle extraction: completed obstacle_positions=%d numpy_borders=%d carla_borders=%d',
        len(obstacle_positions),
        len(numpy_obstacle_borders),
        len(carla_obstacle_borders),
    )

    return obstacle_positions, numpy_obstacle_borders, carla_obstacle_borders


def generate_ellipse_border(transform, extent_x, extent_y, resolution, size_factor=np.sqrt(2)):
    """Generate border points in shape of ellipse"""

    ellipse = []
    circumference = 2 * extent_x + 2 * extent_y
    samples = max([6, int(circumference / resolution)])
    for theta in (np.pi * 2 * i / samples for i in range(samples)):
        point = [extent_x * np.cos(theta) * size_factor, extent_y * np.sin(theta) * size_factor, 0.0]
        point_loc = transform.transform(carla.Location(point[0], point[1], point[2]))

        ellipse.append(point_loc)

    return ellipse


def bb_contains(bounding_box, location, transform):
    """Check if a location is within a bounding box"""

    diff = bounding_box.location - location
    diff = transform.transform(diff)

    x_cond = abs(diff.x) < bounding_box.extent.x
    y_cond = abs(diff.y) < bounding_box.extent.y
    z_cond = abs(diff.z) < bounding_box.extent.z

    return x_cond and y_cond and z_cond


def get_dynamic_obstacles(carla_world, scenario_config):
    """Get dynamic obstacles from CARLA world (vehicles) and generate obstacle border around bounding box"""

    resolution = scenario_config.get('obstacles', {}).get('resolution', 0.1)

    vehicles = carla_world.get_actors().filter("*vehicle*")

    carla_obstacle_borders = []
    actor_ids = []
    obstacle_positions = []
    obstacle_headings = []
    obstacle_velocities = []
    obstacle_extents = []

    for v in vehicles:
        actor_ids.append(v.id)
        bb = v.bounding_box
        transform = v.get_transform()
        center = np.array([transform.location.x, transform.location.y])
        obstacle_headings.append(transform.rotation.yaw)
        carla_velocity = v.get_velocity()
        velocity = np.array([carla_velocity.x, carla_velocity.y])
        obstacle_velocities.append(velocity)

        border_points = generate_ellipse_border(transform, bb.extent.x, bb.extent.y, resolution)
        carla_obstacle_borders.append(border_points)
        obstacle_positions.append(center)
        obstacle_extents.append(np.array([bb.extent.x, bb.extent.y]))

    numpy_obstacle_borders = [np.array([[vec.x, vec.y] for vec in border]) for border in carla_obstacle_borders]

    return (actor_ids, obstacle_positions, obstacle_headings, obstacle_velocities, obstacle_extents,
            numpy_obstacle_borders), carla_obstacle_borders


def extract_borders_from_config(scenario_config):
    """Extract manually defined (straight) obstacle borders from scenario configuration"""

    obstacle_config = scenario_config.get('obstacles')

    obstacles = []
    carla_obstacles = []
    section_info = []
    if obstacle_config is not None:
        borders = obstacle_config.get('borders', [])
        obstacle_resolution = obstacle_config.get('resolution', 0.1)

        for border in borders:
            start_point = np.array(border['start_point'])
            end_point = np.array(border['end_point'])

            samples = int(np.linalg.norm(end_point - start_point) / obstacle_resolution)

            border_line = np.column_stack((np.linspace(start_point[0], end_point[0], samples),
                                           np.linspace(start_point[1], end_point[1], samples)))
            middle_loc = border_line[len(border_line) // 2]
            section_length = len(border_line) * obstacle_resolution

            section_info.append([middle_loc, section_length])
            obstacles.append(border_line)
            carla_obstacles.append([carla.Vector3D(p[0], p[1], 0) for p in border_line])

    return obstacles, section_info, carla_obstacles

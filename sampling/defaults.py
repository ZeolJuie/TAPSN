CITY_OBJECT_LABEL_NAMES = [
    'TrafficLight',
    'TrafficSigns',
    'Poles',
    'Static',
    'Fences',
    'Other',
    'Dynamic',
    'GuardRail',
    'Car',
    'Truck',
    'Bus',
    'Motorcycle',
    'Bicycle',
]

DEFAULT_ACTOR_FILTERS = ['*pedestrian*', '*vehicle*']

SENSOR_OUTPUT_SUBDIRS = {
    'camera.rgb': 'image',
    'camera.depth': 'depth',
    'camera.semantic_segmentation': 'semantic',
    'lidar.ray_cast': 'velodyne',
    'lidar.ray_cast_semantic': 'velodyne_semantic',
    'radar': 'radar',
    'imu': 'imu',
    'gnss': 'gnss',
}

SUPPORTED_SENSOR_TYPES = set(SENSOR_OUTPUT_SUBDIRS)

DEFAULT_SAMPLING_POLICY = {
    'sequence_id': 'auto',
    'output_root': 'artifacts/sequences',
    'reuse_existing_root': False,
    'frame_stride': 10,
    'valid_distance': 40.0,
    'sensor_sync_timeout': 1.0,
    'stop_on_ego_end': True,
    'export': {
        'ego_csv': True,
        'labels_json': True,
        'log_json': True,
        'calibration_json': True,
        'dynamic_actors': True,
        'city_objects': True,
        'simulation_records': True,
        'pedestrian_bev': True,
        'config_snapshots': True,
    },
    'annotation': {
        'actor_filters': DEFAULT_ACTOR_FILTERS,
        'city_object_labels': CITY_OBJECT_LABEL_NAMES,
    },
}

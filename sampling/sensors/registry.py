from sampling.sensors.base import (
    CameraSensor,
    DepthCameraSensor,
    GNSSSensor,
    IMUSensor,
    LidarSensor,
    RadarSensor,
    SemanticCameraSensor,
    SemanticLidarSensor,
)


SENSOR_CLASS_BY_TYPE = {
    'camera.rgb': CameraSensor,
    'camera.depth': DepthCameraSensor,
    'camera.semantic_segmentation': SemanticCameraSensor,
    'lidar.ray_cast': LidarSensor,
    'lidar.ray_cast_semantic': SemanticLidarSensor,
    'radar': RadarSensor,
    'imu': IMUSensor,
    'gnss': GNSSSensor,
}


def create_sampling_sensor(
    world,
    blueprint_library,
    parent_actor,
    sensor_spec,
    sequence_manager,
    frame_stride,
    attach_to_parent=True,
):
    sensor_class = SENSOR_CLASS_BY_TYPE[sensor_spec['type']]
    return sensor_class(
        world,
        blueprint_library,
        parent_actor,
        sensor_spec,
        sequence_manager,
        frame_stride,
        attach_to_parent=attach_to_parent,
    )

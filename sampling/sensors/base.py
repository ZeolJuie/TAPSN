import csv
import math
import queue

import carla
import numpy as np


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def set_blueprint_attr_if_supported(bp, attr_name, attr_value):
    if bp.has_attribute(attr_name):
        bp.set_attribute(attr_name, str(attr_value))


def sensor_transform_from_spec(sensor_spec):
    location = sensor_spec.get('location', [0.0, 0.0, 0.0])
    rotation = sensor_spec.get('rotation', [0.0, 0.0, 0.0])
    return carla.Transform(
        carla.Location(x=float(location[0]), y=float(location[1]), z=float(location[2])),
        carla.Rotation(pitch=float(rotation[0]), yaw=float(rotation[1]), roll=float(rotation[2])),
    )


def clone_transform(transform):
    return carla.Transform(
        carla.Location(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z),
        ),
        carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
            roll=float(transform.rotation.roll),
        ),
    )


def world_transform_from_parent(parent_transform, relative_transform):
    return carla.Transform(
        parent_transform.transform(
            carla.Vector3D(
                x=float(relative_transform.location.x),
                y=float(relative_transform.location.y),
                z=float(relative_transform.location.z),
            )
        ),
        carla.Rotation(
            pitch=float(parent_transform.rotation.pitch + relative_transform.rotation.pitch),
            yaw=float(parent_transform.rotation.yaw + relative_transform.rotation.yaw),
            roll=float(parent_transform.rotation.roll + relative_transform.rotation.roll),
        ),
    )


def projection_matrix(width, height, fov_deg):
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return [
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ]


class BaseSamplingSensor:
    def __init__(self, world, blueprint_library, parent_actor, sensor_spec, sequence_manager, frame_stride, attach_to_parent=True):
        self.world = world
        self.blueprint_library = blueprint_library
        self.parent_actor = parent_actor
        self.sensor_spec = sensor_spec
        self.sequence_manager = sequence_manager
        self.frame_stride = max(int(frame_stride), 1)
        self.name = sensor_spec['name']
        self.type = sensor_spec['type']
        self.output_dir = sequence_manager.sensor_output_dir(sensor_spec)
        self.queue = queue.Queue()
        self.base_relative_tf = sensor_transform_from_spec(sensor_spec)
        self.attach_to_parent = bool(attach_to_parent)
        self.sensor_bp, self.sensor = self._spawn_sensor()

    def _spawn_sensor(self):
        raise NotImplementedError

    def _spawn_sensor_actor(self, blueprint):
        relative_transform = self.relative_transform()
        if self.attach_to_parent:
            return self.world.spawn_actor(blueprint, relative_transform, attach_to=self.parent_actor)
        spawn_transform = world_transform_from_parent(self.parent_actor.get_transform(), relative_transform)
        return self.world.spawn_actor(blueprint, spawn_transform)

    def _save_selected_frame(self, sensor_data):
        raise NotImplementedError

    def start(self):
        self.sensor.listen(self._callback)

    def _callback(self, sensor_data):
        if sensor_data.frame % self.frame_stride != 0:
            return
        self._save_selected_frame(sensor_data)
        self.queue.put((sensor_data.frame, self.name))

    def wait_for_frame(self, frame_id, timeout=1.0):
        while True:
            try:
                queued_frame, _ = self.queue.get(timeout=timeout)
            except queue.Empty:
                return False
            if queued_frame < frame_id:
                continue
            return queued_frame == frame_id

    def relative_transform(self):
        return clone_transform(self.base_relative_tf)

    def set_relative_transform(self, transform):
        if self.attach_to_parent:
            self.sensor.set_transform(transform)
        else:
            self.set_world_transform_from_parent(transform)

    def set_world_transform_from_parent(self, relative_transform):
        parent_transform = self.parent_actor.get_transform()
        self.sensor.set_transform(world_transform_from_parent(parent_transform, relative_transform))

    def calibration_record(self):
        record = {
            'name': self.name,
            'type': self.type,
            'location': list(self.sensor_spec.get('location', [0.0, 0.0, 0.0])),
            'rotation': list(self.sensor_spec.get('rotation', [0.0, 0.0, 0.0])),
            'attributes': {},
        }
        for attribute_name in self.sensor_bp:
            record['attributes'][attribute_name.id] = str(attribute_name)
        return record

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()


class CameraSensor(BaseSamplingSensor):
    blueprint_id = 'sensor.camera.rgb'
    converter = None
    file_extension = 'jpg'

    def _spawn_sensor(self):
        bp = self.blueprint_library.find(self.blueprint_id)
        set_blueprint_attr_if_supported(bp, 'image_size_x', int(self.sensor_spec.get('image_size_x', 1600)))
        set_blueprint_attr_if_supported(bp, 'image_size_y', int(self.sensor_spec.get('image_size_y', 900)))
        set_blueprint_attr_if_supported(bp, 'fov', float(self.sensor_spec.get('fov', 90.0)))
        set_blueprint_attr_if_supported(bp, 'sensor_tick', float(self.sensor_spec.get('sensor_tick', 0.05)))
        return bp, self._spawn_sensor_actor(bp)

    def _save_selected_frame(self, sensor_data):
        file_path = self.output_dir / f'{sensor_data.frame:06d}.{self.file_extension}'
        ensure_parent(file_path)
        if self.converter is not None:
            sensor_data.convert(self.converter)
        sensor_data.save_to_disk(str(file_path))

    def calibration_record(self):
        record = super().calibration_record()
        width = int(self.sensor_spec.get('image_size_x', 1600))
        height = int(self.sensor_spec.get('image_size_y', 900))
        fov = float(self.sensor_spec.get('fov', 90.0))
        record['camera_intrinsic'] = projection_matrix(width, height, fov)
        return record


class DepthCameraSensor(CameraSensor):
    blueprint_id = 'sensor.camera.depth'
    file_extension = 'png'


class SemanticCameraSensor(CameraSensor):
    blueprint_id = 'sensor.camera.semantic_segmentation'
    converter = carla.ColorConverter.CityScapesPalette


class LidarSensor(BaseSamplingSensor):
    blueprint_id = 'sensor.lidar.ray_cast'

    def _spawn_sensor(self):
        bp = self.blueprint_library.find(self.blueprint_id)
        set_blueprint_attr_if_supported(bp, 'dropoff_general_rate', float(self.sensor_spec.get('dropoff_general_rate', 0.0)))
        set_blueprint_attr_if_supported(bp, 'dropoff_intensity_limit', float(self.sensor_spec.get('dropoff_intensity_limit', 1.0)))
        set_blueprint_attr_if_supported(bp, 'dropoff_zero_intensity', float(self.sensor_spec.get('dropoff_zero_intensity', 0.0)))
        set_blueprint_attr_if_supported(bp, 'upper_fov', float(self.sensor_spec.get('upper_fov', 25.0)))
        set_blueprint_attr_if_supported(bp, 'lower_fov', float(self.sensor_spec.get('lower_fov', -25.0)))
        set_blueprint_attr_if_supported(bp, 'channels', int(self.sensor_spec.get('channels', 64)))
        set_blueprint_attr_if_supported(bp, 'range', float(self.sensor_spec.get('range', 40.0)))
        set_blueprint_attr_if_supported(bp, 'rotation_frequency', float(self.sensor_spec.get('rotation_frequency', 20.0)))
        set_blueprint_attr_if_supported(bp, 'points_per_second', int(self.sensor_spec.get('points_per_second', 1000000)))
        set_blueprint_attr_if_supported(bp, 'sensor_tick', float(self.sensor_spec.get('sensor_tick', 0.05)))
        return bp, self._spawn_sensor_actor(bp)

    def _save_selected_frame(self, sensor_data):
        data = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        data[:, 1] = -data[:, 1]

        file_path = self.output_dir / f'{sensor_data.frame:06d}.bin'
        ensure_parent(file_path)
        data.astype(np.float32).tofile(file_path)


class SemanticLidarSensor(LidarSensor):
    blueprint_id = 'sensor.lidar.ray_cast_semantic'

    def _save_selected_frame(self, sensor_data):
        point_dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('cos_angle', np.float32),
            ('obj_idx', np.uint32),
            ('obj_tag', np.uint32),
        ])
        data = np.copy(np.frombuffer(sensor_data.raw_data, dtype=point_dtype))
        data['y'] = -data['y']

        file_path = self.output_dir / f'{sensor_data.frame:06d}.bin'
        ensure_parent(file_path)
        data.astype(point_dtype).tofile(file_path)


class RadarSensor(BaseSamplingSensor):
    def _spawn_sensor(self):
        bp = self.blueprint_library.find('sensor.other.radar')
        set_blueprint_attr_if_supported(bp, 'horizontal_fov', float(self.sensor_spec.get('horizontal_fov', 90.0)))
        set_blueprint_attr_if_supported(bp, 'vertical_fov', float(self.sensor_spec.get('vertical_fov', 30.0)))
        set_blueprint_attr_if_supported(bp, 'points_per_second', int(self.sensor_spec.get('points_per_second', 15000)))
        set_blueprint_attr_if_supported(bp, 'range', float(self.sensor_spec.get('range', 30.0)))
        set_blueprint_attr_if_supported(bp, 'sensor_tick', float(self.sensor_spec.get('sensor_tick', 0.05)))
        return bp, self._spawn_sensor_actor(bp)

    def _save_selected_frame(self, sensor_data):
        data = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
        data = np.reshape(data, (len(sensor_data), 4))

        file_path = self.output_dir / f'{sensor_data.frame:06d}.npy'
        ensure_parent(file_path)
        np.save(file_path, data)


class CsvStreamSensor(BaseSamplingSensor):
    csv_filename = None
    header = ()

    def __init__(self, *args, **kwargs):
        self.csv_path = None
        super().__init__(*args, **kwargs)

    def _initialize_csv(self):
        if self.csv_path is None:
            self.csv_path = self.output_dir / self.csv_filename
            ensure_parent(self.csv_path)
            with self.csv_path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(self.header)

    def _append_row(self, row):
        self._initialize_csv()
        with self.csv_path.open('a', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(row)


class IMUSensor(CsvStreamSensor):
    csv_filename = 'imu.csv'
    header = ('frame', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'compass')

    def _spawn_sensor(self):
        bp = self.blueprint_library.find('sensor.other.imu')
        set_blueprint_attr_if_supported(bp, 'noise_accel_stddev_x', float(self.sensor_spec.get('noise_accel_stddev_x', 0.01)))
        set_blueprint_attr_if_supported(bp, 'noise_accel_stddev_y', float(self.sensor_spec.get('noise_accel_stddev_y', 0.01)))
        set_blueprint_attr_if_supported(bp, 'noise_accel_stddev_z', float(self.sensor_spec.get('noise_accel_stddev_z', 0.01)))
        set_blueprint_attr_if_supported(bp, 'noise_gyro_stddev_x', float(self.sensor_spec.get('noise_gyro_stddev_x', 0.001)))
        set_blueprint_attr_if_supported(bp, 'noise_gyro_stddev_y', float(self.sensor_spec.get('noise_gyro_stddev_y', 0.001)))
        set_blueprint_attr_if_supported(bp, 'noise_gyro_stddev_z', float(self.sensor_spec.get('noise_gyro_stddev_z', 0.001)))
        set_blueprint_attr_if_supported(bp, 'sensor_tick', float(self.sensor_spec.get('sensor_tick', 0.05)))
        return bp, self._spawn_sensor_actor(bp)

    def _save_selected_frame(self, sensor_data):
        self._append_row([
            f'{sensor_data.frame:06d}',
            sensor_data.accelerometer.x,
            sensor_data.accelerometer.y,
            sensor_data.accelerometer.z,
            sensor_data.gyroscope.x,
            sensor_data.gyroscope.y,
            sensor_data.gyroscope.z,
            sensor_data.compass,
        ])


class GNSSSensor(CsvStreamSensor):
    csv_filename = 'gnss.csv'
    header = (
        'frame', 'timestamp', 'gnss_latitude', 'gnss_longitude', 'gnss_altitude',
        'location_x', 'location_y', 'location_z', 'rotation_pitch', 'rotation_roll', 'rotation_yaw',
    )

    def _spawn_sensor(self):
        bp = self.blueprint_library.find('sensor.other.gnss')
        set_blueprint_attr_if_supported(bp, 'noise_alt_stddev', float(self.sensor_spec.get('noise_alt_stddev', 0.2)))
        set_blueprint_attr_if_supported(bp, 'noise_lat_stddev', float(self.sensor_spec.get('noise_lat_stddev', 0.000001)))
        set_blueprint_attr_if_supported(bp, 'noise_lon_stddev', float(self.sensor_spec.get('noise_lon_stddev', 0.000001)))
        set_blueprint_attr_if_supported(bp, 'sensor_tick', float(self.sensor_spec.get('sensor_tick', 0.05)))
        return bp, self._spawn_sensor_actor(bp)

    def _save_selected_frame(self, sensor_data):
        transform = sensor_data.transform
        self._append_row([
            f'{sensor_data.frame:06d}',
            f'{sensor_data.timestamp:.12f}',
            sensor_data.latitude,
            sensor_data.longitude,
            sensor_data.altitude,
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.pitch,
            transform.rotation.roll,
            transform.rotation.yaw,
        ])

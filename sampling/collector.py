import csv
import json
import logging
import math
from datetime import datetime

from sampling.annotators import actor_annotations, city_object_annotations
from sampling.defaults import CITY_OBJECT_LABEL_NAMES, DEFAULT_ACTOR_FILTERS
from sampling.sensors import create_sampling_sensor


class SamplingCollector:
    def __init__(self, carla_sim, sequence_manager):
        self.carla_sim = carla_sim
        self.sequence_manager = sequence_manager
        self.world = carla_sim.world
        self.blueprint_library = carla_sim.blueprint_library
        self.policy = sequence_manager.policy
        self.sensor_rig = sequence_manager.sensor_rig

        self.actor_filters = self.policy.get('annotation', {}).get('actor_filters', DEFAULT_ACTOR_FILTERS)
        self.city_object_labels = self.policy.get('annotation', {}).get('city_object_labels', CITY_OBJECT_LABEL_NAMES)
        self.valid_distance = float(self.policy.get('valid_distance', 40.0))
        self.frame_stride = max(int(self.policy.get('frame_stride', 10)), 1)
        self.sensor_sync_timeout = float(self.policy.get('sensor_sync_timeout', 1.0))

        self.ego_actor = None
        self.ego_metadata = None
        self.sensors = []
        self.sensor_by_name = {}
        self.labels = []
        self.ego_pose_rows = []
        self.log_info = None
        self.frame_count = 0
        self.start_sim_time = None
        self.active = False
        self.gait_motion = self.sensor_rig.get('gait_motion', {})
        self.gait_enabled = bool(self.gait_motion.get('enabled', False))
        self.gait_phase = 0.0
        self.last_motion_time = None
        self.sensor_pose_rows = []
        self.pending_ego_actor_id = None
        self.pending_ego_frame = None
        self.manual_sensor_world_pose = False

    def try_activate(self):
        if self.active:
            return
        if self.sequence_manager.resolved_ego is None:
            return

        snapshot = self.world.get_snapshot()
        current_frame = int(snapshot.frame)
        actor_id = int(self.sequence_manager.resolved_ego['actor_id'])

        if self.pending_ego_actor_id != actor_id:
            self.pending_ego_actor_id = actor_id
            self.pending_ego_frame = current_frame
            logging.info(
                'Pending sampling activation for ego ped=%s actor_id=%d at frame=%d; delaying sensor attach until a later frame.',
                self.sequence_manager.resolved_ego.get('ped_name'),
                actor_id,
                current_frame,
            )
            return

        if self.pending_ego_frame is not None and current_frame <= self.pending_ego_frame:
            return

        ego_actor = self.world.get_actor(actor_id)
        if ego_actor is None:
            return

        self.ego_actor = ego_actor
        self.ego_metadata = dict(self.sequence_manager.resolved_ego)
        self.manual_sensor_world_pose = str(self.ego_actor.type_id).startswith('walker.')
        self._spawn_sensors()
        self._initialize_log()
        self._write_calibration_json()
        self._update_sensor_motion(snapshot)
        self.active = True
        self.pending_ego_actor_id = None
        self.pending_ego_frame = None
        logging.info('Activated sampling collector for ego ped=%s actor_id=%d at frame=%d.',
                     self.ego_metadata.get('ped_name'), actor_id, current_frame)

    def _spawn_sensors(self):
        for sensor_spec in self.sensor_rig.get('sensor', []):
            sensor = create_sampling_sensor(
                self.world,
                self.blueprint_library,
                self.ego_actor,
                sensor_spec,
                self.sequence_manager,
                self.frame_stride,
                attach_to_parent=not self.manual_sensor_world_pose,
            )
            sensor.start()
            self.sensors.append(sensor)
            self.sensor_by_name[sensor.name] = sensor

    def _initialize_log(self):
        walker_bp_id = self.ego_actor.type_id
        map_name = self.world.get_map().name.split('/')[-1]
        self.start_sim_time = self.world.get_snapshot().timestamp.elapsed_seconds
        self.log_info = {
            'sequence_id': self.sequence_manager.root.name,
            'scenario_name': self.sequence_manager.scenario_config.get('scenario_name'),
            'scene_id': self.sequence_manager.scenario_config.get('map', {}).get('scene_id'),
            'location': map_name,
            'date_captured': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'weather': self.carla_sim.generated_weather,
            'walker': {
                'blueprint_id': walker_bp_id,
                'actor_id': int(self.ego_actor.id),
                'ped_name': self.ego_metadata.get('ped_name'),
                'group_id': self.ego_metadata.get('group_id'),
                'group_member_index': self.ego_metadata.get('group_member_index'),
            },
            'frame_num': 0,
            'duration': 0.0,
            'sensor_rig_name': self.sensor_rig.get('name'),
        }

    def _write_calibration_json(self):
        export_cfg = self.policy.get('export', {})
        if not export_cfg.get('calibration_json', True):
            return

        payload = {
            'sequence_id': self.sequence_manager.root.name,
            'ego_actor_id': int(self.ego_actor.id),
            'sensors': [sensor.calibration_record() for sensor in self.sensors],
        }
        path = self.sequence_manager.paths['sampling'] / 'calibration.json'
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')

    def _gait_activity(self, speed):
        speed_min = float(self.gait_motion.get('speed_min', 0.2))
        speed_ref = float(self.gait_motion.get('speed_ref', 1.4))
        if speed <= speed_min:
            return 0.0
        if speed_ref <= speed_min:
            return 1.0
        s = max(0.0, min(1.0, (speed - speed_min) / (speed_ref - speed_min)))
        return s * s * (3.0 - 2.0 * s)

    def _update_sensor_motion(self, snapshot):
        if not self.active or self.ego_actor is None:
            return

        current_time = snapshot.timestamp.elapsed_seconds
        if self.last_motion_time is None:
            self.last_motion_time = current_time
            dt = 0.0
        else:
            dt = max(0.0, current_time - self.last_motion_time)
            self.last_motion_time = current_time

        velocity = self.ego_actor.get_velocity()
        speed = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
        activity = self._gait_activity(speed) if self.gait_enabled else 0.0

        freq_min = float(self.gait_motion.get('frequency_min_hz', 1.2))
        freq_max = float(self.gait_motion.get('frequency_max_hz', 2.2))
        waist_amp_min = float(self.gait_motion.get('waist_z_amplitude_min_m', 0.01))
        waist_amp_max = float(self.gait_motion.get('waist_z_amplitude_max_m', 0.04))
        head_z_scale = float(self.gait_motion.get('head_z_scale', 0.75))
        head_pitch_min = float(self.gait_motion.get('head_pitch_amplitude_min_deg', 0.2))
        head_pitch_max = float(self.gait_motion.get('head_pitch_amplitude_max_deg', 1.2))

        frequency = freq_min + activity * (freq_max - freq_min)
        self.gait_phase += 2.0 * math.pi * frequency * dt

        waist_amp = waist_amp_min + activity * (waist_amp_max - waist_amp_min)
        head_pitch_amp = head_pitch_min + activity * (head_pitch_max - head_pitch_min)

        waist_bob = activity * waist_amp * math.sin(self.gait_phase)
        head_bob = head_z_scale * waist_bob
        head_pitch = activity * head_pitch_amp * math.sin(self.gait_phase + math.pi / 2.0)

        for sensor in self.sensors:
            transform = sensor.relative_transform()
            mount = str(sensor.sensor_spec.get('mount', 'waist')).lower()
            if mount == 'head':
                transform.location.z += head_bob
                transform.rotation.pitch += head_pitch
                gait_z = head_bob
                gait_pitch = head_pitch
            else:
                transform.location.z += waist_bob
                gait_z = waist_bob
                gait_pitch = 0.0
            sensor.set_relative_transform(transform)
            self.sensor_pose_rows.append({
                'frame': f'{snapshot.frame:06d}',
                'sensor_name': sensor.name,
                'mount': mount,
                'relative_location_x': transform.location.x,
                'relative_location_y': transform.location.y,
                'relative_location_z': transform.location.z,
                'relative_rotation_pitch': transform.rotation.pitch,
                'relative_rotation_yaw': transform.rotation.yaw,
                'relative_rotation_roll': transform.rotation.roll,
                'gait_phase': self.gait_phase,
                'gait_z_offset': gait_z,
                'gait_pitch_offset_deg': gait_pitch,
                'ego_speed': speed,
                'activity': activity,
            })

    def update_sensor_world_poses(self):
        if not self.active or not self.manual_sensor_world_pose or self.ego_actor is None:
            return

        for sensor in self.sensors:
            sensor.set_world_transform_from_parent(sensor.relative_transform())

    def capture_frame(self, snapshot):
        if not self.active:
            return

        if self.ego_actor is None or self.ego_actor.is_alive is False:
            return

        frame_id = int(snapshot.frame)
        if frame_id % self.frame_stride != 0:
            return

        for sensor in self.sensors:
            if not sensor.wait_for_frame(frame_id, timeout=self.sensor_sync_timeout):
                logging.warning('Sampling sensor %s missed frame %d.', sensor.name, frame_id)
                return

        self.frame_count += 1
        self.log_info['frame_num'] = self.frame_count

        self._record_ego_pose(snapshot)
        self._record_labels(frame_id)
        self._update_sensor_motion(snapshot)
        self._record_sensor_pose(frame_id)

    def _record_ego_pose(self, snapshot):
        transform = self.ego_actor.get_transform()
        velocity = self.ego_actor.get_velocity()
        logging.info(
            'Sampling ego world pose frame=%06d loc=(%.3f, %.3f, %.3f) rot=(roll=%.3f, pitch=%.3f, yaw=%.3f)',
            int(snapshot.frame),
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.roll,
            transform.rotation.pitch,
            transform.rotation.yaw,
        )
        self.ego_pose_rows.append({
            'frame': f'{snapshot.frame:06d}',
            'timestamp': f'{snapshot.timestamp.elapsed_seconds:.12f}',
            'location_x': transform.location.x,
            'location_y': transform.location.y,
            'location_z': transform.location.z,
            'rotation_roll': transform.rotation.roll,
            'rotation_pitch': transform.rotation.pitch,
            'rotation_yaw': transform.rotation.yaw,
            'velocity_x': velocity.x,
            'velocity_y': velocity.y,
            'velocity_z': velocity.z,
        })

    def _record_labels(self, frame_id):
        export_cfg = self.policy.get('export', {})
        if not export_cfg.get('labels_json', True):
            return

        frame_label = {'frame_id': f'{frame_id:06d}', 'objects': []}
        if export_cfg.get('dynamic_actors', True):
            frame_label['objects'].extend(
                actor_annotations(
                    self.world,
                    self.ego_actor,
                    self.actor_filters,
                    self.valid_distance,
                    pedestrian_metadata_lookup=self.sequence_manager.get_pedestrian_metadata,
                )
            )
        if export_cfg.get('city_objects', True):
            frame_label['objects'].extend(
                city_object_annotations(self.world, self.ego_actor, self.city_object_labels, self.valid_distance)
            )
        self.labels.append(frame_label)

    def _record_sensor_pose(self, frame_id):
        for sensor in self.sensors:
            transform = sensor.sensor.get_transform()
            if sensor.name == 'CAM_FRONT':
                logging.info(
                    'Sampling CAM_FRONT world pose frame=%06d loc=(%.3f, %.3f, %.3f) rot=(roll=%.3f, pitch=%.3f, yaw=%.3f)',
                    int(frame_id),
                    transform.location.x,
                    transform.location.y,
                    transform.location.z,
                    transform.rotation.roll,
                    transform.rotation.pitch,
                    transform.rotation.yaw,
                )
            matrix = transform.get_matrix()
            for row in self.sensor_pose_rows:
                if row['frame'] == f'{frame_id:06d}' and row['sensor_name'] == sensor.name:
                    row.update({
                        'world_location_x': transform.location.x,
                        'world_location_y': transform.location.y,
                        'world_location_z': transform.location.z,
                        'world_rotation_pitch': transform.rotation.pitch,
                        'world_rotation_yaw': transform.rotation.yaw,
                        'world_rotation_roll': transform.rotation.roll,
                        'world_m00': matrix[0][0],
                        'world_m01': matrix[0][1],
                        'world_m02': matrix[0][2],
                        'world_m03': matrix[0][3],
                        'world_m10': matrix[1][0],
                        'world_m11': matrix[1][1],
                        'world_m12': matrix[1][2],
                        'world_m13': matrix[1][3],
                        'world_m20': matrix[2][0],
                        'world_m21': matrix[2][1],
                        'world_m22': matrix[2][2],
                        'world_m23': matrix[2][3],
                        'world_m30': matrix[3][0],
                        'world_m31': matrix[3][1],
                        'world_m32': matrix[3][2],
                        'world_m33': matrix[3][3],
                    })
                    break

    def close(self):
        if self.active:
            self._write_outputs()
        for sensor in self.sensors:
            sensor.destroy()
        self.sensors = []
        self.sensor_by_name = {}
        self.active = False

    def _write_outputs(self):
        sampling_root = self.sequence_manager.paths['sampling']
        export_cfg = self.policy.get('export', {})

        end_sim_time = self.world.get_snapshot().timestamp.elapsed_seconds
        if self.start_sim_time is not None and self.log_info is not None:
            self.log_info['duration'] = end_sim_time - self.start_sim_time

        if export_cfg.get('ego_csv', True):
            path = sampling_root / 'ego.csv'
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    'frame', 'timestamp',
                    'location_x', 'location_y', 'location_z',
                    'rotation_roll', 'rotation_pitch', 'rotation_yaw',
                    'velocity_x', 'velocity_y', 'velocity_z',
                ])
                writer.writeheader()
                writer.writerows(self.ego_pose_rows)

        if export_cfg.get('labels_json', True):
            path = sampling_root / 'labels.json'
            path.write_text(json.dumps(self.labels, indent=2), encoding='utf-8')

        if export_cfg.get('log_json', True) and self.log_info is not None:
            path = sampling_root / 'log.json'
            path.write_text(json.dumps(self.log_info, indent=2, sort_keys=True), encoding='utf-8')

        if self.sensor_pose_rows:
            path = sampling_root / 'sensor_pose.csv'
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    'frame',
                    'sensor_name',
                    'mount',
                    'relative_location_x',
                    'relative_location_y',
                    'relative_location_z',
                    'relative_rotation_pitch',
                    'relative_rotation_yaw',
                    'relative_rotation_roll',
                    'gait_phase',
                    'gait_z_offset',
                    'gait_pitch_offset_deg',
                    'ego_speed',
                    'activity',
                    'world_location_x',
                    'world_location_y',
                    'world_location_z',
                    'world_rotation_pitch',
                    'world_rotation_yaw',
                    'world_rotation_roll',
                    'world_m00',
                    'world_m01',
                    'world_m02',
                    'world_m03',
                    'world_m10',
                    'world_m11',
                    'world_m12',
                    'world_m13',
                    'world_m20',
                    'world_m21',
                    'world_m22',
                    'world_m23',
                    'world_m30',
                    'world_m31',
                    'world_m32',
                    'world_m33',
                ])
                writer.writeheader()
                writer.writerows(self.sensor_pose_rows)

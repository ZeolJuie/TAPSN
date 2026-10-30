import json
import logging
import shutil
import time
from pathlib import Path

from sampling.defaults import SENSOR_OUTPUT_SUBDIRS
from sampling.ego_selector import resolve_declared_ego, matches_declared_ego


REPO_ROOT = Path(__file__).resolve().parents[1]


def _json_dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


class SamplingSequenceManager:
    def __init__(self, scenario_config, sampling_setup):
        self.scenario_config = scenario_config
        self.sampling_setup = sampling_setup
        self.policy = sampling_setup['policy']
        self.sensor_rig = sampling_setup['sensor_rig']
        self.declared_ego = resolve_declared_ego(scenario_config)
        if self.declared_ego is None:
            raise ValueError('sampling.enabled=true requires declaring exactly one ego pedestrian.')
        self.resolved_ego = None
        self.pedestrians_by_name = {}
        self.pedestrians_by_actor_id = {}

        self.sequence_id = self._resolve_sequence_id()
        self.root = self._resolve_sequence_root()
        self.paths = self._build_paths()
        self._ensure_layout()
        self._write_config_snapshots()
        self._write_manifest(status='initialized')

    def _resolve_sequence_id(self):
        configured = self.policy.get('sequence_id', 'auto')
        if configured and configured != 'auto':
            return str(configured)

        scenario_name = self.scenario_config.get('scenario_name', 'sequence')
        time_stamp = time.strftime('%Y%m%d-%H%M%S')
        return f'{time_stamp}-{scenario_name}'

    def _resolve_sequence_root(self):
        base_root = (REPO_ROOT / Path(self.policy.get('output_root', 'artifacts/sequences'))).resolve()
        root = base_root / self.sequence_id
        if self.policy.get('reuse_existing_root', False):
            return root
        if not root.exists():
            return root

        suffix = time.strftime('%Y%m%d-%H%M%S')
        return base_root / f'{self.sequence_id}-{suffix}'

    def _build_paths(self):
        return {
            'config': self.root / 'config',
            'simulation': self.root / 'simulation',
            'visualization': self.root / 'visualization',
            'sampling': self.root / 'sampling',
            'manifest': self.root / 'manifest.json',
            'ego_selection': self.root / 'simulation' / 'ego_selection.json',
            'run_summary': self.root / 'simulation' / 'run_summary.json',
        }

    def _ensure_layout(self):
        for key, path in self.paths.items():
            if key in {'manifest', 'ego_selection', 'run_summary'}:
                continue
            path.mkdir(parents=True, exist_ok=True)

        sampling_root = self.paths['sampling']
        for subdir in sorted(set(SENSOR_OUTPUT_SUBDIRS.values())):
            (sampling_root / subdir).mkdir(parents=True, exist_ok=True)

        for sensor in self.sensor_rig.get('sensor', []):
            sensor_dir = self.sensor_output_dir(sensor)
            if sensor_dir is not None:
                sensor_dir.mkdir(parents=True, exist_ok=True)

    def _write_config_snapshots(self):
        if not self.policy.get('export', {}).get('config_snapshots', True):
            return

        scenario_path = Path(self.sampling_setup['scenario_config_path'])
        self._copy_snapshot_if_needed(scenario_path, self.paths['config'] / 'scenario.toml')

        sensor_rig_path = self.sampling_setup.get('sensor_rig_path')
        if sensor_rig_path:
            self._copy_snapshot_if_needed(Path(sensor_rig_path), self.paths['config'] / 'sensor_rig.toml')

        policy_path = self.sampling_setup.get('policy_path')
        if policy_path:
            self._copy_snapshot_if_needed(Path(policy_path), self.paths['config'] / 'sampling_policy.toml')

        resolved = {
            'policy': self.policy,
            'sensor_rig': self.sensor_rig,
            'declared_ego': self.declared_ego,
        }
        _json_dump(self.paths['config'] / 'sampling_resolved.json', resolved)

    @staticmethod
    def _copy_snapshot_if_needed(source, destination):
        source_path = Path(source).resolve()
        destination_path = Path(destination).resolve()
        if source_path == destination_path:
            return
        shutil.copy2(source_path, destination_path)

    def sensor_output_dir(self, sensor_spec):
        sensor_type = sensor_spec.get('type')
        sensor_name = sensor_spec.get('name')
        sampling_root = self.paths['sampling']
        subdir = SENSOR_OUTPUT_SUBDIRS.get(sensor_type)
        if subdir is None:
            return None

        if sensor_type in {'imu', 'gnss'}:
            return sampling_root / subdir
        return sampling_root / subdir / str(sensor_name)

    def on_ped_spawned(self, ped_name, actor_id, spawn_metadata):
        actor_id = int(actor_id)
        ped_record = {
            'ped_name': ped_name,
            'actor_id': actor_id,
            **spawn_metadata,
        }
        self.pedestrians_by_name[ped_name] = ped_record
        self.pedestrians_by_actor_id[actor_id] = ped_record

        if self.resolved_ego is not None:
            return

        if not matches_declared_ego(self.declared_ego, spawn_metadata):
            return

        self.resolved_ego = dict(ped_record)
        _json_dump(self.paths['ego_selection'], {
            'declared_ego': self.declared_ego,
            'resolved_ego': self.resolved_ego,
        })

    def on_ped_despawned(self, ped_name):
        ped_record = self.pedestrians_by_name.get(ped_name)
        if ped_record is not None:
            ped_record['despawned'] = True

        if self.resolved_ego is None:
            return
        if self.resolved_ego.get('ped_name') != ped_name:
            return
        self.resolved_ego['despawned'] = True
        _json_dump(self.paths['ego_selection'], {
            'declared_ego': self.declared_ego,
            'resolved_ego': self.resolved_ego,
        })

    def get_pedestrian_metadata(self, actor_id):
        return self.pedestrians_by_actor_id.get(int(actor_id))

    def finalize_from_sim_output(self, simulation_output_dir, carla_sim):
        simulation_dir = self.paths['simulation']
        simulation_dir.mkdir(parents=True, exist_ok=True)

        copied_files = []
        if simulation_output_dir is not None:
            for source in Path(simulation_output_dir).iterdir():
                if not source.is_file():
                    continue
                destination = simulation_dir / source.name
                shutil.copy2(source, destination)
                copied_files.append(destination.name)
        else:
            copied_files = sorted(path.name for path in simulation_dir.iterdir() if path.is_file())

        bev_output_path = None
        if self.policy.get('export', {}).get('pedestrian_bev', True):
            bev_output_path = self.render_pedestrian_bev()

        if not self.paths['ego_selection'].exists():
            _json_dump(self.paths['ego_selection'], {
                'declared_ego': self.declared_ego,
                'resolved_ego': self.resolved_ego,
            })

        run_summary = {
            'sequence_id': self.root.name,
            'scenario_name': self.scenario_config.get('scenario_name'),
            'map_name': self.scenario_config.get('map', {}).get('map_name'),
            'scene_id': self.scenario_config.get('map', {}).get('scene_id'),
            'simulation_output_dir': str(simulation_dir.resolve()),
            'copied_simulation_files': copied_files,
            'pedestrian_bev': str(bev_output_path) if bev_output_path is not None else None,
            'weather': carla_sim.generated_weather,
            'declared_ego': self.declared_ego,
            'resolved_ego': self.resolved_ego,
        }
        _json_dump(self.paths['run_summary'], run_summary)
        self._write_manifest(status='finalized')

    def _write_manifest(self, status):
        manifest = {
            'sequence_id': self.root.name,
            'status': status,
            'root': str(self.root),
            'config_dir': str(self.paths['config']),
            'simulation_dir': str(self.paths['simulation']),
            'visualization_dir': str(self.paths['visualization']),
            'sampling_dir': str(self.paths['sampling']),
            'sensor_count': len(self.sensor_rig.get('sensor', [])),
            'declared_ego': self.declared_ego,
            'resolved_ego': self.resolved_ego,
        }
        _json_dump(self.paths['manifest'], manifest)

    def render_pedestrian_bev(self):
        try:
            from sampling import bev
        except Exception as exc:  # pragma: no cover
            logging.warning('Failed to import pedestrian BEV renderer: %s', exc)
            return None

        pedestrian_csv = self.paths['simulation'] / 'pedestrian.csv'
        if not pedestrian_csv.exists():
            return None

        borders_csv = self.paths['simulation'] / 'borders.csv'
        vehicle_csv = self.paths['simulation'] / 'vehicle.csv'
        obstacles_csv = self.paths['simulation'] / 'obstacles.csv'
        ego_selection_json = self.paths['simulation'] / 'ego_selection.json'
        map_config = self.scenario_config.get('map', {})
        scene_id = map_config.get('scene_id')
        catalog_path = map_config.get('scene_catalog_path')
        if catalog_path is not None:
            catalog_path = REPO_ROOT / Path(catalog_path)

        trajectories = bev.load_pedestrian_trajectories(pedestrian_csv)
        vehicle_trajectories = bev.load_vehicle_trajectories(vehicle_csv)
        scene_cell = bev.load_scene_cell(catalog_path, scene_id) if catalog_path and scene_id else None
        if scene_cell is not None:
            focus_extent = bev.extent_from_scene_bounds(scene_cell['bounds'], padding=1.0)
        else:
            focus_extent = bev.trajectory_extent(trajectories, margin=8.0)
        border_polylines = bev.select_polylines_in_bounds(
            bev.load_border_polylines(borders_csv, 2),
            focus_extent,
        )
        obstacle_polygons = bev.select_obstacles_in_bounds(
            bev.load_obstacle_polygons(obstacles_csv),
            focus_extent,
        )
        ego_ped_id = bev.parse_ego_ped_id(ego_selection_json)

        title = self.scenario_config.get('scenario_name') or self.root.name
        svg = bev.render_svg(
            trajectories=trajectories,
            vehicle_trajectories=vehicle_trajectories,
            border_polylines=border_polylines,
            obstacle_polygons=obstacle_polygons,
            scene_cell=scene_cell,
            ego_ped_id=ego_ped_id,
            width=1440,
            height=960,
            title=title,
        )
        output_path = self.paths['visualization'] / 'pedestrian_bev_group.svg'
        output_path.write_text(svg, encoding='utf-8')
        return output_path

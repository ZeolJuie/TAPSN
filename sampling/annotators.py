import math

import carla
import numpy as np

from sampling.defaults import CITY_OBJECT_LABEL_NAMES


def _build_city_object_label_mapping():
    mapping = {}
    for label_name in CITY_OBJECT_LABEL_NAMES:
        label = getattr(carla.CityObjectLabel, label_name, None)
        if label is not None:
            mapping[label_name] = label
    return mapping


CITY_OBJECT_LABEL_MAPPING = _build_city_object_label_mapping()


def center_point(vertices):
    points = np.array([[vertex.x, vertex.y, vertex.z] for vertex in vertices], dtype=float)
    return points.mean(axis=0).tolist()


def actor_class_name(actor):
    if 'walker' in actor.type_id:
        return 'Pedestrian'

    base_type = actor.attributes.get('base_type', '')
    if base_type:
        return base_type.capitalize()
    return 'Car'


def _rotation_radians(rotation):
    return [
        math.radians(rotation.roll),
        math.radians(rotation.pitch),
        math.radians(rotation.yaw),
    ]


def _velocity_vector(actor):
    velocity = actor.get_velocity()
    velocity_vector = [velocity.x, velocity.y, velocity.z]
    speed = float(np.linalg.norm(velocity_vector))
    return velocity_vector, speed


def _actor_attributes(actor):
    return {str(key): value for key, value in actor.attributes.items()}


def _pedestrian_fields(actor, pedestrian_metadata_lookup):
    metadata = None
    if pedestrian_metadata_lookup is not None:
        metadata = pedestrian_metadata_lookup(int(actor.id))

    ped_name = actor.attributes.get('role_name')
    if metadata is not None and metadata.get('ped_name') is not None:
        ped_name = metadata.get('ped_name')

    return {
        'ped_name': ped_name,
        'group_id': int((metadata or {}).get('group_id', -1)),
        'group_member_index': int((metadata or {}).get('group_member_index', 0)),
        'route_id': (metadata or {}).get('route_id'),
        'behavior_name': (metadata or {}).get('behavior_name'),
    }


def actor_annotations(world, ego_actor, actor_filters, valid_distance, pedestrian_metadata_lookup=None):
    ego_location = ego_actor.get_transform().location
    annotations = []
    for actor_filter in actor_filters:
        for actor in world.get_actors().filter(actor_filter):
            if actor.id == ego_actor.id:
                continue

            transform = actor.get_transform()
            if transform.location.distance(ego_location) >= valid_distance:
                continue

            bbox = actor.bounding_box
            verts = bbox.get_world_vertices(transform)
            velocity_vector, speed = _velocity_vector(actor)
            object_label = {
                'object_id': int(actor.id),
                'class': actor_class_name(actor),
                'truncation': 0.0,
                'occlusion': 0,
                'location': center_point(verts),
                'dimensions': [bbox.extent.x * 2.0, bbox.extent.y * 2.0, bbox.extent.z * 2.0],
                'rotation': _rotation_radians(transform.rotation),
                'type_id': actor.type_id,
                'velocity': velocity_vector,
                'speed': speed,
            }

            if 'walker' in actor.type_id:
                object_label.update({
                    'attributes': _actor_attributes(actor),
                    **_pedestrian_fields(actor, pedestrian_metadata_lookup),
                })

            annotations.append(object_label)
    return annotations


def city_object_annotations(world, ego_actor, city_object_labels, valid_distance):
    ego_location = ego_actor.get_transform().location
    annotations = []

    for label_name in city_object_labels:
        label = CITY_OBJECT_LABEL_MAPPING.get(label_name)
        if label is None:
            continue

        for bbox in world.get_level_bbs(label):
            if bbox.location.distance(ego_location) >= valid_distance:
                continue

            verts = bbox.get_world_vertices(carla.Transform())
            annotations.append({
                'object_id': -1,
                'class': label_name,
                'truncation': 0.0,
                'occlusion': 0,
                'location': center_point(verts),
                'dimensions': [bbox.extent.x * 2.0, bbox.extent.y * 2.0, bbox.extent.z * 2.0],
                'rotation': _rotation_radians(bbox.rotation),
            })
    return annotations

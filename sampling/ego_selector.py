class EgoSelectionError(ValueError):
    pass


def _validate_member_index(quantity, member_index, context):
    if member_index is None:
        if quantity == 1:
            return 0
        raise EgoSelectionError(f'{context} must define ego_member_index when quantity > 1.')

    member_index = int(member_index)
    if member_index < 0 or member_index >= quantity:
        raise EgoSelectionError(
            f'{context}.ego_member_index={member_index} is out of range for quantity={quantity}.'
        )
    return member_index


def resolve_declared_ego(scenario_config):
    walker_config = scenario_config.get('walker', {})
    ped_spawners = walker_config.get('ped_spawner') or []

    declared = []
    for spawner_index, spawn_point in enumerate(ped_spawners):
        if not spawn_point.get('ego', False):
            continue

        quantity = int(spawn_point.get('quantity', 1))
        member_index = _validate_member_index(
            quantity,
            spawn_point.get('ego_member_index'),
            f'walker.ped_spawner[{spawner_index}]',
        )
        declared.append({
            'selector': 'spawner_member',
            'spawner_index': spawner_index,
            'member_index': member_index,
        })

    sampling_ego = (scenario_config.get('sampling') or {}).get('ego')
    if sampling_ego is not None:
        selector = sampling_ego.get('selector', 'spawner_member')
        if selector != 'spawner_member':
            raise EgoSelectionError(f'Unsupported sampling.ego.selector {selector!r}.')

        spawner_index = sampling_ego.get('spawner_index')
        if spawner_index is None:
            raise EgoSelectionError('sampling.ego.spawner_index is required.')

        member_index = int(sampling_ego.get('member_index', 0))
        declared.append({
            'selector': selector,
            'spawner_index': int(spawner_index),
            'member_index': member_index,
        })

    if len(declared) > 1:
        raise EgoSelectionError('Exactly one ego pedestrian must be declared across walker.ped_spawner and sampling.ego.')

    return declared[0] if declared else None


def matches_declared_ego(declared_ego, spawn_metadata):
    if declared_ego is None:
        return False

    return (
        int(spawn_metadata.get('spawner_index', -1)) == int(declared_ego['spawner_index'])
        and int(spawn_metadata.get('spawn_member_index', -1)) == int(declared_ego['member_index'])
    )

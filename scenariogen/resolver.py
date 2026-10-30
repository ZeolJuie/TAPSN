"""Bind a scene plan to concrete simulator resources.

The resolver expands one ScenePlan with count N into N ResolvedCase objects.
Conditions stated in the plan are preserved. Factors marked as any are sampled
per case so the batch varies inside the allowed design space.

All randomness is driven by one seeded generator, so the same plan, catalogs,
seed, and existing case indices produce the same cases.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set

from scenariogen.catalog import Catalog, SceneBinding
from scenariogen.models import ActorSpec, ResolvedCase, RouteSpec, ScenePlan
from scenariogen.schema import (
    CROSSING_BEHAVIORS,
    WALKING_BEHAVIORS,
    weather_presets_for_environment,
)


MAX_PEDESTRIANS = 16
MAX_SUPPORT_SPAWNERS = 15

PEDESTRIAN_TOTAL_RANGES = {
    "sparse": (2, 4),
    "medium": (5, 10),
    "crowded": (11, 16),
    "any": (3, 10),
}

VEHICLE_DENSITY_OPTIONS = ("none", "low", "medium", "high")
VEHICLE_DENSITY_WEIGHTS = (0.10, 0.30, 0.40, 0.20)

GROUP_PATTERN_OPTIONS = ("none", "mixed", "abreast", "follow")
GROUP_PATTERN_WEIGHTS = (0.10, 0.50, 0.20, 0.20)

FLOW_OPTIONS = ("one_way", "two_way")
FLOW_WEIGHTS = (0.50, 0.50)

BEHAVIOR_MIX_OPTIONS = ("walking", "crossing", "mixed")
BEHAVIOR_MIX_WEIGHTS = (0.30, 0.30, 0.40)

GROUP_SPAWN_INTERVAL_S = 0.15


class ResolveError(ValueError):
    """Raised when no catalog can bind the requested scene plan."""


def resolve_cases(
    plan: ScenePlan,
    catalogs: Sequence[Catalog],
    *,
    seed: int = 2026,
    sampling_enabled: bool = True,
    existing_indices: Optional[Dict[str, int]] = None,
) -> List[ResolvedCase]:
    """Expand one ScenePlan into plan.count concrete cases."""

    if not catalogs:
        raise ResolveError("at least one catalog is required")
    if plan.count < 1:
        raise ResolveError(f"plan count must be positive, got {plan.count}")

    rng = random.Random(int(seed))
    used_scenes: Dict[str, Set[str]] = {catalog.map_name: set() for catalog in catalogs}
    case_counts: Dict[str, int] = dict(existing_indices or {})

    resolved: List[ResolvedCase] = []
    for _ in range(plan.count):
        catalog = _choose_catalog(plan, catalogs, used_scenes, case_counts, rng)
        binding = catalog.bind(
            scene_kind=plan.topology.get("scene_kind", "any"),
            route_types=plan.topology.get("route_types") or (),
            ego_behavior=plan.ego.get("behavior", "any"),
            excluded_scene_ids=used_scenes[catalog.map_name],
            rng=rng,
        )
        used_scenes[catalog.map_name].add(binding.scene_id)

        case_index = int(case_counts.get(catalog.map_name, 0)) + 1
        case_counts[catalog.map_name] = case_index
        case_seed = int(seed) + len(resolved) + 1

        resolved.append(
            _build_case(
                plan=plan,
                catalog=catalog,
                binding=binding,
                case_index=case_index,
                case_seed=case_seed,
                sampling_enabled=sampling_enabled,
                rng=random.Random(case_seed),
            )
        )

    return resolved


def _choose_catalog(
    plan: ScenePlan,
    catalogs: Sequence[Catalog],
    used_scenes: Dict[str, Set[str]],
    case_counts: Dict[str, int],
    rng: random.Random,
) -> Catalog:
    constraints = {
        "scene_kind": plan.topology.get("scene_kind", "any"),
        "route_types": plan.topology.get("route_types") or (),
        "ego_behavior": plan.ego.get("behavior", "any"),
    }

    eligible = [
        catalog
        for catalog in catalogs
        if catalog.has_eligible(
            **constraints,
            excluded_scene_ids=used_scenes.get(catalog.map_name, set()),
        )
    ]

    if not eligible:
        for scene_ids in used_scenes.values():
            scene_ids.clear()
        eligible = [
            catalog
            for catalog in catalogs
            if catalog.has_eligible(**constraints)
        ]

    if not eligible:
        raise ResolveError(
            "no catalog satisfies "
            f"scene_kind={constraints['scene_kind']}, "
            f"route_types={list(constraints['route_types'])}, "
            f"ego_behavior={constraints['ego_behavior']}"
        )

    fewest_cases = min(case_counts.get(catalog.map_name, 0) for catalog in eligible)
    pool = [
        catalog
        for catalog in eligible
        if case_counts.get(catalog.map_name, 0) == fewest_cases
    ]
    return rng.choice(pool)


def _build_case(
    *,
    plan: ScenePlan,
    catalog: Catalog,
    binding: SceneBinding,
    case_index: int,
    case_seed: int,
    sampling_enabled: bool,
    rng: random.Random,
) -> ResolvedCase:
    ego_behavior = _resolve_ego_behavior(plan, binding, rng)
    density = _resolve_any(
        plan.social.get("pedestrian_density"),
        ("sparse", "medium", "crowded"),
        (0.30, 0.50, 0.20),
        rng,
    )
    group_pattern = _resolve_any(
        plan.social.get("group_pattern"),
        GROUP_PATTERN_OPTIONS,
        GROUP_PATTERN_WEIGHTS,
        rng,
    )
    flow = _resolve_any(plan.social.get("flow"), FLOW_OPTIONS, FLOW_WEIGHTS, rng)
    behavior_mix = _resolve_any(
        plan.social.get("behavior_mix"),
        BEHAVIOR_MIX_OPTIONS,
        BEHAVIOR_MIX_WEIGHTS,
        rng,
    )

    support_actors = _build_support_actors(
        binding=binding,
        density=density,
        group_pattern=group_pattern,
        flow=flow,
        behavior_mix=behavior_mix,
        rng=rng,
    )
    _assign_spawn_times(support_actors, rng)

    ego = _build_ego_actor(
        binding=binding,
        behavior=ego_behavior,
        support_actors=support_actors,
        sampling_enabled=sampling_enabled,
        rng=rng,
    )
    actors = support_actors + [ego]

    return ResolvedCase(
        case_index=case_index,
        case_id=f"{case_index:03d}",
        plan=plan,
        map_name=catalog.map_name,
        catalog_path=catalog.catalog_path,
        scene_id=binding.scene_id,
        scene_kind=binding.scene_kind,
        main_route=binding.main_route,
        support_routes=list(binding.support_routes),
        actors=actors,
        weather_presets=_resolve_weather_presets(plan, rng),
        vehicle_density=_resolve_vehicle_density(plan, rng),
        sampling_enabled=sampling_enabled,
        seed=case_seed,
    )


def _resolve_ego_behavior(plan: ScenePlan, binding: SceneBinding, rng: random.Random) -> str:
    specified = plan.ego.get("behavior", "any")
    if specified not in (None, "", "any"):
        return specified
    return rng.choice(binding.main_route.supported_behaviors)


def _resolve_any(value, options: Sequence[str], weights: Sequence[float], rng: random.Random) -> str:
    if value in (None, "", "any"):
        return rng.choices(options, weights=weights, k=1)[0]
    return value


def _resolve_weather_presets(plan: ScenePlan, rng: random.Random) -> Dict[str, float]:
    presets = weather_presets_for_environment(plan.environment)
    if not presets:
        return {}
    chosen = rng.choice(sorted(presets))
    return {chosen: 1.0}


def _resolve_vehicle_density(plan: ScenePlan, rng: random.Random) -> str:
    specified = plan.traffic.get("vehicle_density", "any")
    if specified not in (None, "", "any"):
        return specified
    return rng.choices(
        VEHICLE_DENSITY_OPTIONS,
        weights=VEHICLE_DENSITY_WEIGHTS,
        k=1,
    )[0]


def _build_support_actors(
    *,
    binding: SceneBinding,
    density: str,
    group_pattern: str,
    flow: str,
    behavior_mix: str,
    rng: random.Random,
) -> List[ActorSpec]:
    actors: List[ActorSpec] = []
    group_needed = group_pattern in {"mixed", "abreast", "follow"}
    group_created = False
    two_way_done = False

    target_total = _target_pedestrians(density, group_needed, rng)
    remaining = max(target_total - 1, 0)

    while remaining > 0 and len(actors) < MAX_SUPPORT_SPAWNERS:
        routes = _candidate_routes(binding, flow, behavior_mix, two_way_done)
        if not routes:
            break

        route = rng.choice(routes)
        behavior = _choose_behavior(route, behavior_mix, rng)
        if behavior is None:
            break

        make_group = False
        if remaining >= 2 and group_pattern != "none":
            if not group_created and group_needed:
                make_group = True
            elif rng.random() < 0.35:
                make_group = True

        if make_group:
            quantity = 3 if remaining >= 3 and rng.random() < 0.35 else 2
            formation = (
                group_pattern
                if group_pattern in {"abreast", "follow"}
                else rng.choice(["abreast", "follow"])
            )
            group_created = True
        else:
            quantity = 1
            formation = None

        actors.append(
            ActorSpec(
                role="support_group" if quantity > 1 else "support_single",
                route_id=route.route_id,
                route_type=route.route_type,
                behavior=behavior,
                quantity=quantity,
                formation=formation,
                spawn_interval=GROUP_SPAWN_INTERVAL_S if quantity > 1 else None,
            )
        )
        remaining -= quantity

        if flow == "two_way" and _is_reverse(route.route_id) != _is_reverse(binding.main_route.route_id):
            two_way_done = True

    return actors


def _target_pedestrians(density: str, group_needed: bool, rng: random.Random) -> int:
    low, high = PEDESTRIAN_TOTAL_RANGES.get(density, PEDESTRIAN_TOTAL_RANGES["any"])
    if group_needed:
        low = max(low, 3)
    return min(rng.randint(low, high), MAX_PEDESTRIANS)


def _candidate_routes(
    binding: SceneBinding,
    flow: str,
    behavior_mix: str,
    two_way_done: bool,
) -> List[RouteSpec]:
    routes = list(binding.support_routes) or [binding.main_route]
    main_is_reverse = _is_reverse(binding.main_route.route_id)

    if flow == "one_way":
        same_direction = [route for route in routes if _is_reverse(route.route_id) == main_is_reverse]
        routes = same_direction or routes

    if flow == "two_way" and not two_way_done:
        opposite = [route for route in routes if _is_reverse(route.route_id) != main_is_reverse]
        if opposite:
            routes = opposite

    candidates = [route for route in routes if _behaviors_for_mix(route, behavior_mix)]
    return candidates or routes


def _behaviors_for_mix(route: RouteSpec, behavior_mix: str) -> List[str]:
    if behavior_mix == "walking":
        return [behavior for behavior in route.supported_behaviors if behavior in WALKING_BEHAVIORS]
    if behavior_mix == "crossing":
        return [behavior for behavior in route.supported_behaviors if behavior in CROSSING_BEHAVIORS]
    return list(route.supported_behaviors)


def _choose_behavior(route: RouteSpec, behavior_mix: str, rng: random.Random) -> Optional[str]:
    behaviors = _behaviors_for_mix(route, behavior_mix) or list(route.supported_behaviors)
    if not behaviors:
        return None
    return rng.choice(behaviors)


def _assign_spawn_times(actors: List[ActorSpec], rng: random.Random) -> None:
    if not actors:
        return

    step = 8.0 / len(actors)
    for index, actor in enumerate(actors):
        jitter = rng.uniform(0.0, min(0.5, step * 0.4))
        actor.spawn_time = round(min(8.0, index * step + jitter), 2)

    actors.sort(key=lambda actor: actor.spawn_time)


def _build_ego_actor(
    *,
    binding: SceneBinding,
    behavior: str,
    support_actors: List[ActorSpec],
    sampling_enabled: bool,
    rng: random.Random,
) -> ActorSpec:
    if support_actors:
        latest = max(actor.spawn_time for actor in support_actors)
        spawn_time = round(min(latest + rng.uniform(1.0, 5.0), 12.0), 2)
    else:
        spawn_time = 0.0

    return ActorSpec(
        role="ego",
        route_id=binding.main_route.route_id,
        route_type=binding.main_route.route_type,
        behavior=behavior,
        quantity=1,
        spawn_time=spawn_time,
        ego=bool(sampling_enabled),
        ego_member_index=0 if sampling_enabled else None,
    )


def _is_reverse(route_id: str) -> bool:
    return str(route_id).endswith("__reverse")

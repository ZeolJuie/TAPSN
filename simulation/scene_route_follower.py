from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


SCENE_ROUTE_MIN_LOOKAHEAD_M = 3.0
SCENE_ROUTE_MAX_LOOKAHEAD_M = 6.0
SCENE_ROUTE_LOOKAHEAD_TIME_S = 2.0
SCENE_ROUTE_CROSSING_APPROACH_M = 3.0
SCENE_ROUTE_CROSSING_ENTRY_MARGIN_M = 1.0
SCENE_ROUTE_POST_CROSSING_LOOKAHEAD_M = 0.75
SCENE_ROUTE_COMPLETION_MARGIN_M = 1.0
SCENE_ROUTE_PROGRESS_BACKTRACK_TOLERANCE_M = 2.0
SCENE_ROUTE_PROGRESS_TRAVEL_BUFFER_M = 0.8
SCENE_ROUTE_MAX_PROGRESS_STEP_M = 1.25


@dataclass(frozen=True)
class NavigationTarget:
    next_waypoint: np.ndarray
    crossing_road: bool
    complete: bool
    progress_s: float
    route_point: np.ndarray
    route_tangent: np.ndarray
    route_force_active: bool


class SceneRoute:
    def __init__(self, polyline_points: Sequence[Sequence[float]], segment_crossing_bools: Sequence[bool]):
        if len(polyline_points) < 2:
            raise ValueError('SceneRoute requires at least two polyline points.')

        points = [np.array(point, dtype=float) for point in polyline_points]
        crossing_bools = list(segment_crossing_bools)
        if len(crossing_bools) != len(points) - 1:
            raise ValueError('SceneRoute segment crossing flags must match polyline segments.')

        filtered_points = [points[0]]
        filtered_crossing_bools = []
        for next_point, crossing_road in zip(points[1:], crossing_bools):
            if np.linalg.norm(next_point[:2] - filtered_points[-1][:2]) <= 1e-6:
                continue
            filtered_points.append(next_point)
            filtered_crossing_bools.append(bool(crossing_road))

        if len(filtered_points) < 2:
            raise ValueError('SceneRoute polyline collapsed to fewer than two unique points.')

        self.points = np.array(filtered_points, dtype=float)
        self.segment_crossing_bools = filtered_crossing_bools
        self.segment_vectors = np.diff(self.points, axis=0)
        self.segment_lengths = np.linalg.norm(self.segment_vectors[:, :2], axis=1)
        self.cumulative_lengths = np.concatenate(([0.0], np.cumsum(self.segment_lengths)))
        self.total_length = float(self.cumulative_lengths[-1])

    def point_at(self, progress_s: float) -> np.ndarray:
        clamped_progress = float(np.clip(progress_s, 0.0, self.total_length))
        if clamped_progress >= self.total_length:
            return np.array(self.points[-1], dtype=float, copy=True)

        segment_index = self.segment_index_at(clamped_progress)
        segment_start = self.points[segment_index]
        segment_end = self.points[segment_index + 1]
        segment_progress = clamped_progress - self.cumulative_lengths[segment_index]
        segment_length = self.segment_lengths[segment_index]
        if segment_length <= 1e-6:
            return np.array(segment_end, dtype=float, copy=True)

        interpolation_ratio = segment_progress / segment_length
        return segment_start + (segment_end - segment_start) * interpolation_ratio

    def segment_index_at(self, progress_s: float) -> int:
        if progress_s >= self.total_length:
            return len(self.segment_lengths) - 1

        search_progress = max(0.0, progress_s)
        segment_index = int(np.searchsorted(self.cumulative_lengths[1:], search_progress, side='right'))
        return min(segment_index, len(self.segment_lengths) - 1)

    def tangent_at(self, progress_s: float) -> np.ndarray:
        segment_index = self.segment_index_at(progress_s)
        segment_vector = np.array(self.segment_vectors[segment_index], dtype=float, copy=True)
        tangent_xy = segment_vector[:2]
        norm = float(np.linalg.norm(tangent_xy))
        if norm <= 1e-6:
            tangent_xy = np.array([1.0, 0.0], dtype=float)
        else:
            tangent_xy = tangent_xy / norm
        return np.array([tangent_xy[0], tangent_xy[1], 0.0], dtype=float)

    def project_progress(self, location: np.ndarray, min_progress_s: float = 0.0) -> float:
        candidate_indices = []
        for segment_index, (segment_start, segment_vector, segment_length) in enumerate(
            zip(self.points[:-1], self.segment_vectors, self.segment_lengths)
        ):
            if segment_length <= 1e-6:
                continue

            segment_unit = segment_vector[:2] / segment_length
            location_delta = location[:2] - segment_start[:2]
            interpolation_ratio = float(np.clip(np.dot(location_delta, segment_unit) / segment_length, 0.0, 1.0))
            progress_s = self.cumulative_lengths[segment_index] + interpolation_ratio * segment_length
            if progress_s >= min_progress_s - SCENE_ROUTE_PROGRESS_BACKTRACK_TOLERANCE_M:
                candidate_indices.append(segment_index)

        if not candidate_indices:
            candidate_indices = list(range(len(self.segment_lengths)))

        best_progress = min_progress_s
        best_distance = None
        for segment_index in candidate_indices:
            segment_start = self.points[segment_index]
            segment_vector = self.segment_vectors[segment_index]
            segment_length = self.segment_lengths[segment_index]
            if segment_length <= 1e-6:
                continue

            segment_unit = segment_vector[:2] / segment_length
            location_delta = location[:2] - segment_start[:2]
            interpolation_ratio = float(np.clip(np.dot(location_delta, segment_unit) / segment_length, 0.0, 1.0))
            projection = segment_start + segment_vector * interpolation_ratio
            projection_distance = float(np.linalg.norm(location[:2] - projection[:2]))
            progress_s = self.cumulative_lengths[segment_index] + interpolation_ratio * segment_length

            if best_distance is None or projection_distance < best_distance:
                best_distance = projection_distance
                best_progress = progress_s

        return float(np.clip(best_progress, 0.0, self.total_length))

    def crossing_block_bounds(self, progress_s: float, approach_distance: float) -> Optional[Tuple[float, float]]:
        current_segment_index = self.segment_index_at(progress_s)
        search_limit = min(self.total_length, progress_s + approach_distance)

        for segment_index in range(current_segment_index, len(self.segment_crossing_bools)):
            segment_start_s = self.cumulative_lengths[segment_index]
            if segment_start_s > search_limit:
                break

            if not self.segment_crossing_bools[segment_index]:
                continue

            end_segment_index = segment_index
            while (
                end_segment_index + 1 < len(self.segment_crossing_bools)
                and self.segment_crossing_bools[end_segment_index + 1]
            ):
                end_segment_index += 1
            return float(segment_start_s), float(self.cumulative_lengths[end_segment_index + 1])

        return None


class SceneRouteFollower:
    def __init__(self, route: SceneRoute, spawn_location: Sequence[float]):
        self.route = route
        self.last_location = np.array(spawn_location, dtype=float)
        self.progress_s = route.project_progress(self.last_location)

    @staticmethod
    def _lookahead_distance(target_speed: float) -> float:
        return float(
            np.clip(
                target_speed * SCENE_ROUTE_LOOKAHEAD_TIME_S,
                SCENE_ROUTE_MIN_LOOKAHEAD_M,
                SCENE_ROUTE_MAX_LOOKAHEAD_M,
            )
        )

    def update(self, location: Sequence[float], target_speed: float) -> NavigationTarget:
        location = np.array(location, dtype=float)
        projected_progress = self.route.project_progress(location, min_progress_s=self.progress_s)
        travel_distance = float(np.linalg.norm(location[:2] - self.last_location[:2]))
        progress_limit = self.progress_s + max(
            SCENE_ROUTE_MAX_PROGRESS_STEP_M,
            travel_distance + SCENE_ROUTE_PROGRESS_TRAVEL_BUFFER_M,
        )
        self.progress_s = max(self.progress_s, min(projected_progress, progress_limit))
        self.last_location = np.array(location, dtype=float, copy=True)
        route_point = self.route.point_at(self.progress_s)
        route_tangent = self.route.tangent_at(self.progress_s)

        remaining_distance = self.route.total_length - self.progress_s
        if remaining_distance <= SCENE_ROUTE_COMPLETION_MARGIN_M:
            return NavigationTarget(
                next_waypoint=np.array(self.route.points[-1], dtype=float, copy=True),
                crossing_road=False,
                complete=True,
                progress_s=self.progress_s,
                route_point=route_point,
                route_tangent=route_tangent,
                route_force_active=False,
            )

        lookahead_distance = self._lookahead_distance(target_speed)
        crossing_block = self.route.crossing_block_bounds(
            self.progress_s,
            lookahead_distance + SCENE_ROUTE_CROSSING_APPROACH_M,
        )
        if crossing_block is not None:
            crossing_block_start, crossing_block_end = crossing_block
            if self.progress_s + SCENE_ROUTE_CROSSING_ENTRY_MARGIN_M < crossing_block_start:
                target_progress = min(crossing_block_start, self.progress_s + lookahead_distance)
                return NavigationTarget(
                    next_waypoint=self.route.point_at(target_progress),
                    crossing_road=False,
                    complete=False,
                    progress_s=self.progress_s,
                    route_point=route_point,
                    route_tangent=route_tangent,
                    route_force_active=False,
                )

            target_progress = min(
                self.route.total_length,
                crossing_block_end + SCENE_ROUTE_POST_CROSSING_LOOKAHEAD_M,
            )
            return NavigationTarget(
                next_waypoint=self.route.point_at(target_progress),
                crossing_road=True,
                complete=False,
                progress_s=self.progress_s,
                route_point=route_point,
                route_tangent=route_tangent,
                route_force_active=True,
            )

        target_progress = min(self.route.total_length, self.progress_s + lookahead_distance)
        return NavigationTarget(
            next_waypoint=self.route.point_at(target_progress),
            crossing_road=False,
            complete=False,
            progress_s=self.progress_s,
            route_point=route_point,
            route_tangent=route_tangent,
            route_force_active=False,
        )

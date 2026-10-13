#!/usr/bin/env python3
# habitat_extensions/measures.py
#
# Custom VLN evaluation measures for habitat-lab v0.3.x.
# Registers OracleSuccess, OracleNavigationError, and PL for R2R evaluation.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
from hydra.core.config_store import ConfigStore

from habitat.core.dataset import Episode
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.config.default_structured_configs import MeasurementConfig
from habitat.tasks.nav.nav import DistanceToGoal, Success


@dataclass
class OracleSuccessMeasurementConfig(MeasurementConfig):
    type: str = "OracleSuccess"
    success_distance: float = 3.0


@dataclass
class OracleNavigationErrorMeasurementConfig(MeasurementConfig):
    type: str = "OracleNavigationError"


@dataclass
class PLMeasurementConfig(MeasurementConfig):
    type: str = "PL"


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.oracle_success",
    group="habitat/task/measurements",
    name="oracle_success",
    node=OracleSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.oracle_navigation_error",
    group="habitat/task/measurements",
    name="oracle_navigation_error",
    node=OracleNavigationErrorMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.pl",
    group="habitat/task/measurements",
    name="pl",
    node=PLMeasurementConfig,
)


def euclidean_distance(
    pos_a: Union[List[float], np.ndarray], pos_b: Union[List[float], np.ndarray]
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


@registry.register_measure
class OracleSuccess(Measure):
    cls_uuid: str = "oracle_success"

    def __init__(
        self,
        sim: Simulator,
        config: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._sim = sim
        self._config = config
        self._success_distance = getattr(config, "success_distance", 3.0)
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self, episode: Any, task: EmbodiedTask, *args: Any, **kwargs: Any
    ) -> None:
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0.0
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode: Any, task: EmbodiedTask, *args: Any, **kwargs: Any
    ) -> None:
        if self._metric == 1.0:
            return
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = float(distance_to_target < self._success_distance)


@registry.register_measure
class OracleNavigationError(Measure):
    cls_uuid: str = "oracle_navigation_error"

    def __init__(
        self,
        sim: Simulator,
        config: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self, episode: Any, task: EmbodiedTask, *args: Any, **kwargs: Any
    ) -> None:
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = float("inf")
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode: Any, task: EmbodiedTask, *args: Any, **kwargs: Any
    ) -> None:
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = min(self._metric, distance_to_target)


@registry.register_measure
class PL(Measure):
    """Path length normalized by shortest start-goal distance."""

    def __init__(
        self, sim: Simulator, config: Any, *args: Any, **kwargs: Any
    ) -> None:
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "pl"

    def reset_metric(
        self, episode: Episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ) -> None:
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )
        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode: Episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ) -> None:
        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position
        self._metric = self._start_end_episode_distance / max(
            self._start_end_episode_distance, self._agent_episode_distance
        )



"""Experiment configs (``configs/experiments/<name>.yaml``) and their expansion into run arms.

An experiment has a ``base`` arm plus either
* ``vary``: one factor at a time (plan protocol rule 1). Each value of each factor gives one arm
  that differs from ``base`` in exactly that field; or
* ``grid``: a full factorial over the listed factors (E6 interaction).
Explicit ``arms`` may be listed too. ``frozen: true`` (set at a gate) unlocks the test split.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hgrag.config import CONFIG_DIR, load_yaml


class ArmSpec(BaseModel):
    """Everything that may differ between runs of an experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline: str = "v4"
    model: str | None = None  # generator model key
    chunker: str | None = None
    embedder: str | None = None
    reranker: str | None = None
    extractor: str | None = None
    graph_backend: str | None = None
    context_token_budget: int | None = None

    def differs_from(self, other: ArmSpec) -> list[str]:
        return [f for f in type(self).model_fields if getattr(self, f) != getattr(other, f)]

    def label(self, base: ArmSpec) -> str:
        changed = self.differs_from(base)
        return ",".join(f"{f}={getattr(self, f)}" for f in changed) or "reference"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    frozen: bool = False  # set true at the gate review; unlocks the test split
    datasets: list[str] = Field(default_factory=list)  # empty = every evaluation dataset
    split: str = "dev"
    limit: int | None = None
    base: ArmSpec = Field(default_factory=ArmSpec)
    vary: dict[str, list[Any]] = Field(default_factory=dict)
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    arms: list[ArmSpec] = Field(default_factory=list)

    def expand(self) -> list[ArmSpec]:
        """All arms, de-duplicated, base first."""
        arms: list[ArmSpec] = [self.base]
        for factor, values in self.vary.items():
            arms += [self.base.model_copy(update={factor: value}) for value in values]
        if self.grid:
            factors = list(self.grid)
            for combo in itertools.product(*(self.grid[f] for f in factors)):
                arms.append(self.base.model_copy(update=dict(zip(factors, combo, strict=True))))
        arms += self.arms
        unique: list[ArmSpec] = []
        for arm in arms:
            if arm not in unique:
                unique.append(arm)
        return unique if (self.vary or self.grid or self.arms) else [self.base]

    def violations(self) -> list[str]:
        """Protocol rule 1: every ``vary`` arm differs from ``base`` in exactly one field."""
        problems = []
        for factor in self.vary:
            if factor not in ArmSpec.model_fields:
                problems.append(f"unknown factor '{factor}'")
        for arm in self.expand():
            changed = arm.differs_from(self.base)
            if len(changed) > 1 and not self.grid and arm not in self.arms:
                problems.append(f"arm {arm.label(self.base)} changes {changed}")
        return problems


def load_experiment(name: str, config_dir: Path = CONFIG_DIR) -> ExperimentConfig:
    """Load the experiment file if it exists; an unknown experiment is an unfrozen ad-hoc one."""
    path = config_dir / "experiments" / f"{name}.yaml"
    data = load_yaml(path) if path.is_file() else {}
    return ExperimentConfig(name=name, **data)


def list_experiments(config_dir: Path = CONFIG_DIR) -> list[str]:
    return sorted(p.stem for p in (config_dir / "experiments").glob("*.yaml"))

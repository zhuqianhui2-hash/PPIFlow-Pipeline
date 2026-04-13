from __future__ import annotations

from .base import Step
from .gen import GenStep
from .interface_enrich import InterfaceEnrichStep
from .partial_flow import PartialFlowStep
from .dockq import DockQStep
from .external import (
    SeqDesignStep,
    FlowPackerStep,
    AF3ScoreStep,
    AF3RefoldStep,
)
from .rosetta_steps import RosettaInterfaceStep, RosettaRelaxStep
from .rank import RankStep, RankFeaturesStep

STEP_REGISTRY = {
    "gen": GenStep,
    "seq1": SeqDesignStep,
    "seq2": SeqDesignStep,
    "flowpacker1": FlowPackerStep,
    "flowpacker2": FlowPackerStep,
    "af3score1": AF3ScoreStep,
    "rosetta_interface": RosettaInterfaceStep,
    "interface_enrich": InterfaceEnrichStep,
    "partial": PartialFlowStep,
    "af3score2": AF3ScoreStep,
    "relax": RosettaRelaxStep,
    "dockq": DockQStep,
    "rank_features": RankFeaturesStep,
    "rank_finalize": RankStep,
    "af3_refold": AF3RefoldStep,
}

STEP_ORDER = [
    "gen",
    "seq1",
    "flowpacker1",
    "af3score1",
    "rosetta_interface",
    "interface_enrich",
    "partial",
    "seq2",
    "flowpacker2",
    "af3score2",
    "relax",
    "af3_refold",
    "dockq",
    "rank_features",
    "rank_finalize",
]

__all__ = ["STEP_REGISTRY", "STEP_ORDER", "Step"]

"""Preference-pair synthesis for DPO."""

from tiny_llm_pipeline.pref.synthesize import (
    CHOSEN_SYS,
    REJECTED_SYS,
    chosen_system,
    is_rejected,
    synth_one,
    synth_prefs,
)

__all__ = [
    "CHOSEN_SYS",
    "REJECTED_SYS",
    "chosen_system",
    "is_rejected",
    "synth_one",
    "synth_prefs",
]

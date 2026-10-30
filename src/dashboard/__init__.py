"""Componentes de apresentação do F1 Data Lab (apenas UI e adaptações de dados)."""

from .data import (
    RAW_DIR,
    PRESENTATION_REVISION,
    file_signature,
    load_app_data,
    load_audit_report,
    evaluate_model_experiment,
    apply_explorer_filters,
    distinct_grid_options,
    race_options,
    driver_options,
    constructor_options,
    scope_summary,
)
from .theme import apply_theme, plotly_layout, scope_note

__all__ = [
    "RAW_DIR",
    "PRESENTATION_REVISION",
    "file_signature",
    "load_app_data",
    "load_audit_report",
    "evaluate_model_experiment",
    "apply_explorer_filters",
    "distinct_grid_options",
    "race_options",
    "driver_options",
    "constructor_options",
    "scope_summary",
    "apply_theme",
    "plotly_layout",
    "scope_note",
]

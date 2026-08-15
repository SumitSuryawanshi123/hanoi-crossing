"""Hanoi Crossing: a two-player, partially-observed Tower of Hanoi variant.

The public API re-exported here is the core engine surface — see
``hanoi_crossing.engine`` for the full implementation and docstrings.
"""

from hanoi_crossing.engine import (
    Action,
    ActionResult,
    ActionType,
    ALL_ACTIONS,
    GameState,
    PlayerState,
    apply_action,
    create_initial_state,
    legal_actions,
    legal_mask,
    observe,
)

__all__ = [
    "Action",
    "ActionResult",
    "ActionType",
    "ALL_ACTIONS",
    "GameState",
    "PlayerState",
    "apply_action",
    "create_initial_state",
    "legal_actions",
    "legal_mask",
    "observe",
]

"""JSON I/O helpers shared by the replay and random-play CLIs.

This module is deliberately kept separate from :mod:`hanoi_crossing.engine`:
it is frontend glue (parsing/formatting), not part of the "core engine"
line-count budget.
"""

import json
from typing import Any

from hanoi_crossing.engine import (
    Action,
    ActionType,
    GameState,
    PlayerId,
)

_ACTION_TYPES = {t.value: t for t in ActionType}


def parse_action(entry: dict) -> Action:
    """Parse ``{"action": "lift"|"place"|"skip", "pole": 1|2|3|null}`` into an Action."""
    if "action" not in entry:
        raise ValueError(f"turn entry missing 'action': {entry!r}")
    raw_type = entry["action"]
    if raw_type not in _ACTION_TYPES:
        raise ValueError(f"unknown action type {raw_type!r}; expected one of {sorted(_ACTION_TYPES)}")
    return Action(type=_ACTION_TYPES[raw_type], pole=entry.get("pole"))


def parse_turn(entry: dict) -> tuple[PlayerId, Action]:
    """Parse one ``{"player": "A"|"B", "action": ..., "pole": ...}`` turn entry."""
    if "player" not in entry:
        raise ValueError(f"turn entry missing 'player': {entry!r}")
    player = entry["player"]
    if player not in ("A", "B"):
        raise ValueError(f"turn entry has invalid player {player!r}; expected 'A' or 'B'")
    return player, parse_action(entry)


def load_game_file(path: str) -> tuple[int, list[tuple[PlayerId, Action]]]:
    """Load a replay-format game file: ``{"n_disks": int, "turns": [...]}``."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "n_disks" not in data:
        raise ValueError("game file missing required field 'n_disks'")
    n_disks = data["n_disks"]
    if not isinstance(n_disks, int) or isinstance(n_disks, bool) or n_disks < 0:
        raise ValueError(f"'n_disks' must be a non-negative integer, got {n_disks!r}")
    if "turns" not in data or not isinstance(data["turns"], list):
        raise ValueError("game file missing required list field 'turns'")
    turns = [parse_turn(entry) for entry in data["turns"]]
    return n_disks, turns


def action_to_dict(action: Action) -> dict:
    return {"action": action.type.value, "pole": action.pole}


def player_state_to_dict(state: GameState, player: PlayerId) -> dict:
    own = state.players[player]
    return {"pole1": list(own.pole1), "pole3": list(own.pole3), "hand": own.hand}


def state_to_dict(state: GameState) -> dict:
    """Full (God's-eye) final-state snapshot, for CLI output."""
    return {
        "n_disks": state.n_disks,
        "shared": list(state.shared),
        "players": {p: player_state_to_dict(state, p) for p in ("A", "B")},
        "winners": sorted(state.winners),
        "move_count": state.move_count,
    }


def turn_log_entry(
    index: int,
    player: PlayerId,
    action: Action,
    *,
    legal: bool | None = None,
    reason: str | None = None,
    winners_after: frozenset[PlayerId] = frozenset(),
    skipped: bool = False,
) -> dict:
    entry: dict[str, Any] = {"index": index, "player": player, **action_to_dict(action)}
    if skipped:
        entry["skipped"] = True
        return entry
    entry["legal"] = legal
    if reason is not None:
        entry["reason"] = reason
    if winners_after:
        entry["winners_after"] = sorted(winners_after)
    return entry


def dump_json(data: dict, output_path: str | None = None, pretty: bool = False) -> None:
    """Write ``data`` as JSON to ``output_path``, or print to stdout if omitted."""
    indent = 2 if pretty else None
    text = json.dumps(data, indent=indent)
    if output_path is None:
        print(text)
    else:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")


def turns_to_record(turns: list[tuple[PlayerId, Action]], n_disks: int) -> dict:
    """Build a replay-format ``{"n_disks", "turns"}`` document from taken turns."""
    return {
        "n_disks": n_disks,
        "turns": [{"player": p, **action_to_dict(a)} for p, a in turns],
    }

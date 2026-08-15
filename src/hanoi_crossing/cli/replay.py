"""Replay CLI: read pre-recorded moves + turn order, output the final state.

Usage:
    uv run hanoi-replay path/to/game.json [--output out.json] [--pretty]

Input schema (see README for full details)::

    {
      "n_disks": 1,
      "turns": [
        {"player": "A", "action": "lift", "pole": 1},
        {"player": "B", "action": "lift", "pole": 1},
        {"player": "A", "action": "place", "pole": 3}
      ]
    }
"""

import argparse
import sys

from hanoi_crossing.engine import Action, GameState, PlayerId, apply_action, create_initial_state, has_winner
from hanoi_crossing.serialization import dump_json, load_game_file, state_to_dict, turn_log_entry


def run_replay(
    n_disks: int, turns: list[tuple[PlayerId, Action]]
) -> tuple[GameState, list[dict]]:
    """Apply ``turns`` in order against a fresh game, stopping once any
    player has won. Returns the final state and a per-turn log.

    Stopping at the first win is a driver-level policy (the engine itself
    has no opinion on when an episode ends): once ``state.winners`` is
    non-empty, remaining turns are recorded as ``skipped`` rather than
    applied.
    """
    state = create_initial_state(n_disks)
    log: list[dict] = []
    for index, (player, action) in enumerate(turns):
        if has_winner(state):
            log.append(turn_log_entry(index, player, action, skipped=True))
            continue
        result = apply_action(state, player, action)
        state = result.state
        log.append(
            turn_log_entry(
                index,
                player,
                action,
                legal=result.legal,
                reason=result.reason,
                winners_after=result.winners,
            )
        )
    return state, log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hanoi-replay",
        description="Replay a pre-recorded Hanoi Crossing game and print the final state.",
    )
    parser.add_argument("input", help="path to a game JSON file ({n_disks, turns})")
    parser.add_argument("--output", "-o", default=None, help="write JSON result here instead of stdout")
    parser.add_argument("--pretty", action="store_true", help="pretty-print the JSON output")
    args = parser.parse_args(argv)

    try:
        n_disks, turns = load_game_file(args.input)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    final_state, log = run_replay(n_disks, turns)

    output = state_to_dict(final_state)
    output["turns"] = log
    try:
        dump_json(output, args.output, args.pretty)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

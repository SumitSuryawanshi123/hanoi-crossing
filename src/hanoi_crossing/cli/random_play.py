"""Random-play mode: both players make random *legal* moves.

Usage:
    uv run hanoi-random --n-disks 3 --max-turns 300 --seed 42 \\
        [--turn-policy alternating|random] [--record path.json] [--output out.json] [--pretty]

This frontend deliberately consumes the engine exactly as an external
agent would: on each turn it calls :func:`hanoi_crossing.engine.legal_actions`
(which is itself derived only from :func:`hanoi_crossing.engine.observe`,
the partially-observed view) and picks uniformly at random among the
legal actions. It never inspects hidden opponent state.
"""

import argparse
import random
import sys

from hanoi_crossing.engine import (
    Action,
    GameState,
    PlayerId,
    apply_action,
    create_initial_state,
    has_winner,
    legal_actions,
)
from hanoi_crossing.serialization import (
    dump_json,
    state_to_dict,
    turn_log_entry,
    turns_to_record,
)

PLAYERS: tuple[PlayerId, PlayerId] = ("A", "B")


def run_random_play(
    n_disks: int,
    max_turns: int,
    seed: int | None,
    turn_policy: str = "alternating",
) -> tuple[GameState, list[dict], list[tuple[PlayerId, Action]], str]:
    """Play a random game and return ``(final_state, turn_log, taken_turns, termination_reason)``.

    ``termination_reason`` is ``"winner"`` or ``"max_turns_reached"`` — a
    driver-level concept; the engine itself has no notion of episode
    termination.
    """
    if turn_policy not in ("alternating", "random"):
        raise ValueError(f"unknown turn_policy {turn_policy!r}; expected 'alternating' or 'random'")

    rng = random.Random(seed)
    state = create_initial_state(n_disks)
    log: list[dict] = []
    taken: list[tuple[PlayerId, Action]] = []
    reason = "max_turns_reached"

    for index in range(max_turns):
        if has_winner(state):
            reason = "winner"
            break

        player = PLAYERS[index % 2] if turn_policy == "alternating" else rng.choice(PLAYERS)
        action = rng.choice(legal_actions(state, player))

        result = apply_action(state, player, action)
        state = result.state
        taken.append((player, action))
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
    else:
        if has_winner(state):
            reason = "winner"

    return state, log, taken, reason


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hanoi-random",
        description="Play a random-valid-moves Hanoi Crossing game and print the final state.",
    )
    parser.add_argument("--n-disks", type=int, default=3, help="disks per player (default: 3)")
    parser.add_argument("--max-turns", type=int, default=500, help="safety cap on turns (default: 500)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed, for reproducibility")
    parser.add_argument(
        "--turn-policy",
        choices=("alternating", "random"),
        default="alternating",
        help="who acts each turn: strict A/B alternation, or a coin flip (default: alternating)",
    )
    parser.add_argument("--record", default=None, help="also write a replay-compatible turns file here")
    parser.add_argument("--output", "-o", default=None, help="write JSON result here instead of stdout")
    parser.add_argument("--pretty", action="store_true", help="pretty-print the JSON output")
    args = parser.parse_args(argv)

    try:
        final_state, log, taken, reason = run_random_play(
            args.n_disks, args.max_turns, args.seed, args.turn_policy
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output = state_to_dict(final_state)
    output["termination_reason"] = reason
    output["turns"] = log
    try:
        dump_json(output, args.output, args.pretty)
        if args.record is not None:
            dump_json(turns_to_record(taken, args.n_disks), args.record, args.pretty)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

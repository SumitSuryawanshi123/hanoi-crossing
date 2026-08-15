# Hanoi Crossing

A two-player, partially-observed Tower of Hanoi variant, implemented as a
reusable Python game engine plus two thin CLI frontends: **replay** (feed
it pre-recorded moves, get the final state) and **random-play** (both
players sample uniformly among their legal moves).

```
        1a
        |
 1b -- [2] -- 3b
        |
        3a
```

Player A sees poles `1a – 2 – 3a`. Player B sees poles `1b – 2 – 3b`. Pole
`2` is shared: both players can see it and either may lift/place disks on
it. Neither player can see the other's poles 1/3, nor what the other holds
in hand.

## Project layout

```
hanoi-crossing/
  pyproject.toml
  README.md
  src/
    hanoi_crossing/
      engine.py          # core engine: model + rules (211 lines, no I/O)
      serialization.py   # JSON <-> engine objects (frontend glue)
      cli/
        replay.py         # `hanoi-replay` entry point
        random_play.py    # `hanoi-random` entry point
  tests/
    test_engine.py        # exercises the engine directly
    test_random_play.py
    test_replay_cli.py
    fixtures/example_n1.json
```

Standard `src/`-layout package managed with [uv](https://docs.astral.sh/uv/).

## Running it

```bash
uv sync                                 # create the venv, install deps
uv run pytest -v                        # run the test suite

uv run hanoi-replay tests/fixtures/example_n1.json --pretty
uv run hanoi-random --n-disks 3 --seed 42 --pretty
```

(If you don't have `uv`, a plain `pip install -e .` into a venv works too,
since everything is standard `pyproject.toml`/hatchling.)

## Internal model

`src/hanoi_crossing/engine.py` is the entire core engine — **211 lines**,
well under the 500-line constraint — and has zero dependencies outside the
standard library. It exposes a small set of plain dataclasses and pure
functions:

- `PlayerState`: one player's private `pole1`, `pole3` (stacks, bottom-to-top),
  and `hand` (`int | None`).
- `GameState`: `n_disks`, the shared `pole2` stack, a `{player_id: PlayerState}`
  dict, a sticky `winners: frozenset[PlayerId]`, and a `move_count`.
- `Action`: `{type: LIFT|PLACE|SKIP, pole: 1|2|3|None}`. `pole` is always
  relative to the *acting* player — `1`/`3` mean "my own pole", `2` always
  means the shared pole. There is no way to construct an action that
  references the opponent's pole 1 or 3, so private-pole visibility is
  enforced structurally by the action schema itself, not by a runtime check.
- `ALL_ACTIONS`: the fixed 7-action space (3 lifts + 3 places + skip),
  exposed so a caller (e.g. an RL policy head) can rely on a constant-size
  discrete action space regardless of `n_disks`.

Core functions, all pure (`state in, new-state out` — nothing is mutated
in place, nothing lives at module scope):

| Function | Purpose |
|---|---|
| `create_initial_state(n_disks)` | Deal odds to A, evens to B, largest at the bottom. |
| `observe(state, player)` | The player's **partial observation**: own `pole1`/`pole2`/`pole3`/`hand`, nothing about the opponent. |
| `legal_actions(state, player)` / `legal_mask(state, player)` | Legal moves right now, computed *only* from `observe()`. |
| `apply_action(state, player, action)` | Applies one action; returns an `ActionResult(state, legal, reason, winners)`. |
| `has_winner(state)` | `bool(state.winners)` convenience helper. |

## Why the engine looks like this (reuse beyond the two frontends)

The brief asks for an engine that could later become the core of an RL
training loop or a concurrent simulation service, unchanged. Two decisions
make that possible:

1. **No turn-order state at all.** `apply_action` takes the acting player
   as an explicit argument on every call; the engine never tracks "whose
   turn it is". Turn order is entirely a caller concern (a fixed list for
   replay, an alternating/random policy for random-play, an RL loop's own
   scheduler, or a service dispatching whichever player's client just sent
   a request). This is a literal reading of "the engine must not assume
   any particular turn-order pattern" — it doesn't just avoid assuming a
   *specific* pattern, it has no turn-order concept whatsoever.
2. **Immutable-style state transitions.** `apply_action` never mutates its
   input; it returns a brand-new `GameState`. A concurrent service can hold
   thousands of independent `GameState` values with no risk of one game's
   step accidentally touching another's memory, and can snapshot/rewind
   states trivially (useful for RL replay buffers).
3. **`legal_actions`/`legal_mask` are derived only from `observe()`.** This
   is what lets the random-play frontend "consume the engine exactly as an
   external agent would": it never reaches into `GameState` fields it
   shouldn't see. An RL policy wired up the same way is automatically
   respecting partial observability, by construction.

## Design decisions (where the rules were open to interpretation)

The rules were fairly precise, but a few points needed an explicit choice.
Each is implemented and covered by a test in `tests/test_engine.py`.

1. **Disk ownership is implicit, never stored.** All `2N` disk sizes are
   globally unique and their parity never changes (odd ⇒ dealt to A,
   even ⇒ dealt to B), so there's no need for a separate "owner" field —
   not even for disks sitting on the shared pole. Placement legality only
   ever depends on size, matching "on top of a strictly larger disk"
   literally (ownership is irrelevant to the placement rule).

2. **What exactly is illegal.** Enumerated explicitly rather than left
   fuzzy:
   - `LIFT` with a non-empty hand, or from an empty pole.
   - `PLACE` with an empty hand, or onto a pole whose top disk is not
     strictly larger than the held disk.
   - Any action naming a pole outside `{1, 2, 3}` fails at `Action`
     construction time (`ValueError`) — this is treated as a *malformed
     request*, not a wasted turn, since it isn't a meaningful in-game
     choice at all. This keeps CLI input validation (syntax) cleanly
     separate from game-rule legality (semantics): `serialization.py`
     parses and validates JSON shape and raises early with a clear CLI
     error; `engine.apply_action` only ever needs to reason about
     well-formed `Action`s.
   - All other illegal actions simply waste the turn: the returned state
     is identical to the input in every field except `move_count` (bumped
     so a caller can still budget/limit wasted turns).

3. **Skip is always legal** and is a genuine no-op (bumps `move_count`
   only) — useful when a player has no move they want to make, and a
   natural "pass" action for a random or RL agent.

4. **Win is (re-)checked for *both* players after every applied action**,
   not just the mover's — because clearing the shared pole (the only piece
   of state either player can affect on the *other's* behalf) can be the
   move that completes the *other* player's win. Concretely: `winners` is
   a `frozenset`, and `apply_action` unions in every player whose win
   condition newly became true this call.

   A closer look shows this can still only add **one new winner per call**
   in legitimate play: `PLACE` never touches the shared pole (so it can't
   complete the *other* player's win), and the only action that shrinks
   the shared pole — `LIFT(2)` — always leaves the mover's own hand
   occupied (so it can never complete the mover's *own* win in that same
   call). So while both players ending up in `winners` is common (finish
   your own stack, then eventually the other player finishes theirs), it
   is *discovered* across two separate turns, not atomically — and
   `winners` accumulating without ever forgetting a past winner reflects
   that.  `apply_action`'s union logic itself still supports reporting
   several new winners from one call (tested directly against a
   hand-built state in `test_win_check_unions_multiple_newly_satisfied_players_defensively`),
   in case a future rule tweak ever made simultaneous wins reachable.

5. **The engine never halts on a win.** `state.winners` is sticky (a
   player, once recorded as a winner, stays recorded even if they later
   move disks again), but nothing in `engine.py` stops `apply_action` from
   being called again afterwards — ending an episode is a driver decision
   (an RL loop might want a few extra steps of look-ahead, a live service
   might want to let the losing player keep fiddling). Both reference
   CLIs *do* stop feeding further turns once any player has won — see
   `run_replay`/`run_random_play` — but that's a frontend policy, not an
   engine rule.

6. **Turn order is external and can be *anything*.** Neither CLI assumes
   strict alternation is required by the engine (only random-play's own
   *default* turn-policy happens to alternate). `test_engine.py` includes
   a test where player A acts three times in a row with no complaint from
   the engine.

7. **`n_disks = 0` is allowed** (both players start and stay empty-handed
   with nothing to move — `legal_actions` degenerates to `[SKIP]`); this
   is just a boundary case that the general formulas naturally support,
   not a special case in the code.

8. **Malformed *external* input never crashes with a raw traceback.** This
   is distinct from in-game illegality (point 2): a wrong-typed or negative
   `n_disks`, an unknown `action` string, a missing `player`/`pole` key, or
   a failure to write `--output`/`--record` (e.g. a bad path) are all
   syntactic/environmental problems, not in-game moves, so they're validated
   in `serialization.load_game_file`/`parse_turn` or caught around the
   output-writing calls in both CLIs' `main()`, and reported as
   `error: ...` on stderr with exit code `1`. `Action.__post_init__` also
   explicitly rejects `bool` for `pole` (since `True == 1`/`False == 0` in
   Python, a naive `pole not in (1, 2, 3)` check would silently accept
   `pole=True` as "pole 1").

## Input / output formats

### Replay input (`hanoi-replay <file>`)

A single JSON document merges the disk count, the turn order, *and* the
moves — since each turn entry already names its own actor, a separate
"turn order" array would just be redundant:

```json
{
  "n_disks": 1,
  "turns": [
    {"player": "A", "action": "lift", "pole": 1},
    {"player": "B", "action": "lift", "pole": 1},
    {"player": "A", "action": "place", "pole": 3}
  ]
}
```

- `action` is one of `"lift"`, `"place"`, `"skip"`.
- `pole` is `1`, `2`, or `3` (required for `lift`/`place`; omit or `null`
  for `skip`), always relative to `player`.

This is exactly the worked example from the problem statement, saved at
[`tests/fixtures/example_n1.json`](tests/fixtures/example_n1.json).

### Output (both CLIs)

```json
{
  "n_disks": 1,
  "shared": [],
  "players": {
    "A": {"pole1": [], "pole3": [1], "hand": null},
    "B": {"pole1": [], "pole3": [2], "hand": null}
  },
  "winners": ["A"],
  "move_count": 3,
  "turns": [
    {"index": 0, "player": "A", "action": "lift", "pole": 1, "legal": true},
    {"index": 1, "player": "B", "action": "lift", "pole": 1, "legal": true},
    {"index": 2, "player": "A", "action": "place", "pole": 3, "legal": true, "winners_after": ["A"]}
  ]
}
```

- `winners` is the final, God's-eye state — appropriate for an offline
  replay/analysis tool, even though neither in-game player would see it.
- Each turn log entry carries `legal`; when an action is illegal, `reason`
  explains why (`"legal": false, "reason": "cannot lift: pole 3 is empty"`);
  when a turn wasn't applied at all because a winner already existed, it's
  marked `"skipped": true` instead.
- `hanoi-random` additionally reports `"termination_reason"`: `"winner"` or
  `"max_turns_reached"`.

### Random-play recording (`hanoi-random --record path.json`)

Writes a document in the exact same shape as the replay input format
(`{"n_disks", "turns"}`, without the `legal`/`winners_after` annotations),
so a random game can be fed straight back into `hanoi-replay` and
reproduce an identical final state — this is exercised by
[`tests/test_random_play.py::test_recorded_random_game_replays_to_the_same_final_state`](tests/test_random_play.py).

## CLI reference

```
hanoi-replay <input.json> [--output PATH] [--pretty]

hanoi-random [--n-disks N] [--max-turns N] [--seed N]
             [--turn-policy alternating|random]
             [--record PATH] [--output PATH] [--pretty]
```

`--turn-policy` exists mostly to demonstrate turn-order agnosticism: pass
`random` and the actor for each turn is itself a coin flip rather than
strict A/B alternation, and the engine neither notices nor cares.

The random player selects an action by calling `legal_actions(state, player)`
(itself derived only from `observe()`) and drawing uniformly from it with a
seeded `random.Random` — i.e. it plays exactly the way an external,
partially-observed agent (a human, a scripted bot, or an RL policy) would
have to.

## Testing

`uv run pytest -v` runs everything. All three test modules import and call
`hanoi_crossing.engine`/`hanoi_crossing.cli.*` functions directly — there
is no subprocess/CLI-only testing, per the "tests should exercise the
engine directly" requirement (`test_replay_cli.py` does call the CLI's
`main()`, but as a plain Python function call, so failures surface as
normal assertions/tracebacks rather than opaque subprocess output).

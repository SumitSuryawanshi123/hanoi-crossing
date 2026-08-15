"""Tests for the random-play frontend, driving the engine directly."""

import pytest

from hanoi_crossing.cli.random_play import main, run_random_play
from hanoi_crossing.cli.replay import run_replay
from hanoi_crossing.serialization import parse_turn, turns_to_record
from hanoi_crossing.engine import legal_actions


def test_same_seed_is_deterministic():
    r1 = run_random_play(3, max_turns=300, seed=123, turn_policy="alternating")
    r2 = run_random_play(3, max_turns=300, seed=123, turn_policy="alternating")
    state1, log1, taken1, reason1 = r1
    state2, log2, taken2, reason2 = r2
    assert taken1 == taken2
    assert log1 == log2
    assert reason1 == reason2
    assert state1.players["A"].pole1 == state2.players["A"].pole1
    assert state1.winners == state2.winners


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("turn_policy", ["alternating", "random"])
def test_random_play_terminates_cleanly_across_seeds(seed, turn_policy):
    state, log, taken, reason = run_random_play(3, max_turns=500, seed=seed, turn_policy=turn_policy)
    assert reason in ("winner", "max_turns_reached")
    if reason == "winner":
        assert state.winners
    assert len(taken) == len(log)
    assert len(taken) <= 500


def test_random_player_never_takes_an_illegal_action():
    """The reference random player samples only from legal_actions, so
    every logged turn it takes must have been legal at the time."""
    _, log, _, _ = run_random_play(4, max_turns=500, seed=7, turn_policy="random")
    for entry in log:
        assert entry.get("skipped") is not True
        assert entry["legal"] is True


def test_recorded_random_game_replays_to_the_same_final_state():
    n_disks = 3
    rand_state, _, taken, _ = run_random_play(n_disks, max_turns=300, seed=99, turn_policy="alternating")

    # Round-trip through the exact JSON-shaped record, as hanoi-random --record
    # would write it and hanoi-replay would read it.
    record = turns_to_record(taken, n_disks)
    reconstructed_turns = [parse_turn(entry) for entry in record["turns"]]

    replay_state, _ = run_replay(n_disks, reconstructed_turns)

    assert replay_state.players["A"].pole1 == rand_state.players["A"].pole1
    assert replay_state.players["A"].pole3 == rand_state.players["A"].pole3
    assert replay_state.players["B"].pole1 == rand_state.players["B"].pole1
    assert replay_state.players["B"].pole3 == rand_state.players["B"].pole3
    assert replay_state.shared == rand_state.shared
    assert replay_state.winners == rand_state.winners
    assert replay_state.move_count == rand_state.move_count


def test_invalid_turn_policy_raises():
    with pytest.raises(ValueError):
        run_random_play(2, max_turns=10, seed=1, turn_policy="bogus")


def test_negative_n_disks_raises():
    with pytest.raises(ValueError):
        run_random_play(-1, max_turns=10, seed=1)


def test_zero_disks_never_produces_a_winner():
    """With nothing to move, legal_actions degenerates to [SKIP] for both
    players; the game must run out the clock without crashing or winning."""
    state, log, taken, reason = run_random_play(0, max_turns=20, seed=1)
    assert reason == "max_turns_reached"
    assert state.winners == frozenset()
    assert len(taken) == 20
    assert all(entry["action"] == "skip" for entry in log)


def test_cli_reports_error_on_negative_n_disks(capsys):
    exit_code = main(["--n-disks", "-1"])
    assert exit_code == 1
    assert "error" in capsys.readouterr().err.lower()


def test_cli_reports_error_on_unwritable_output_path(tmp_path, capsys):
    bad_output = tmp_path / "no-such-dir" / "out.json"
    exit_code = main(["--n-disks", "2", "--seed", "1", "--max-turns", "20", "--output", str(bad_output)])
    assert exit_code == 1
    assert "error" in capsys.readouterr().err.lower()


def test_cli_reports_error_on_unwritable_record_path(tmp_path, capsys):
    bad_record = tmp_path / "no-such-dir" / "record.json"
    exit_code = main(["--n-disks", "2", "--seed", "1", "--max-turns", "20", "--record", str(bad_record)])
    assert exit_code == 1
    assert "error" in capsys.readouterr().err.lower()


def test_random_player_uses_only_the_partial_observation():
    """Sanity check that legal_actions (what the random player samples
    from) is derivable purely from observe() and never touches the
    opponent's private state — see engine.legal_actions docstring."""
    from hanoi_crossing.engine import create_initial_state

    state = create_initial_state(2)
    actions_before = legal_actions(state, "A")
    # Mutating B's private poles must not affect A's legal actions.
    state.players["B"].pole1 = []
    state.players["B"].pole3 = [99]
    state.players["B"].hand = 42
    actions_after = legal_actions(state, "A")
    assert actions_before == actions_after

"""Tests for the replay CLI, still exercising the engine underneath."""

import json
from pathlib import Path

import pytest

from hanoi_crossing.cli.replay import main, run_replay
from hanoi_crossing.serialization import load_game_file, parse_turn

FIXTURE = Path(__file__).parent / "fixtures" / "example_n1.json"


def test_load_game_file_parses_fixture():
    n_disks, turns = load_game_file(str(FIXTURE))
    assert n_disks == 1
    assert len(turns) == 3
    assert turns[0][0] == "A"


def test_run_replay_matches_readme_example():
    n_disks, turns = load_game_file(str(FIXTURE))
    state, log = run_replay(n_disks, turns)
    assert state.winners == frozenset({"A"})
    assert state.players["A"].pole3 == [1]
    assert len(log) == 3
    assert all(entry["legal"] for entry in log)


def test_cli_main_prints_json_with_winner(capsys):
    exit_code = main([str(FIXTURE)])
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["winners"] == ["A"]
    assert data["players"]["A"]["pole3"] == [1]
    assert len(data["turns"]) == 3


def test_cli_writes_output_file(tmp_path):
    out_path = tmp_path / "result.json"
    exit_code = main([str(FIXTURE), "--output", str(out_path), "--pretty"])
    assert exit_code == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["winners"] == ["A"]


def test_cli_stops_applying_turns_after_a_winner(tmp_path):
    game = {
        "n_disks": 1,
        "turns": [
            {"player": "A", "action": "lift", "pole": 1},
            {"player": "B", "action": "lift", "pole": 1},
            {"player": "A", "action": "place", "pole": 3},
            {"player": "B", "action": "place", "pole": 3},  # would legally end in a B win too
        ],
    }
    path = tmp_path / "game.json"
    path.write_text(json.dumps(game), encoding="utf-8")

    n_disks, turns = load_game_file(str(path))
    state, log = run_replay(n_disks, turns)
    assert state.winners == frozenset({"A"})
    assert log[-1]["skipped"] is True


def test_cli_reports_error_on_malformed_input(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"turns": []}), encoding="utf-8")  # missing n_disks
    exit_code = main([str(path)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_cli_reports_error_on_missing_file():
    exit_code = main(["/no/such/file.json"])
    assert exit_code == 1


def _write_and_run(tmp_path, data, name="game.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_game_file(str(path))


def test_wrong_typed_n_disks_raises_cleanly(tmp_path):
    """A string/float/null n_disks must raise ValueError (not TypeError),
    so the CLI's `except (OSError, ValueError)` can actually catch it."""
    with pytest.raises(ValueError):
        _write_and_run(tmp_path, {"n_disks": "3", "turns": []})
    with pytest.raises(ValueError):
        _write_and_run(tmp_path, {"n_disks": None, "turns": []})
    with pytest.raises(ValueError):
        _write_and_run(tmp_path, {"n_disks": 2.5, "turns": []})


def test_negative_n_disks_raises_cleanly(tmp_path):
    with pytest.raises(ValueError):
        _write_and_run(tmp_path, {"n_disks": -1, "turns": []})


def test_cli_reports_error_on_wrong_typed_n_disks(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"n_disks": "3", "turns": []}), encoding="utf-8")
    exit_code = main([str(path)])
    assert exit_code == 1
    assert "error" in capsys.readouterr().err.lower()


def test_cli_reports_error_on_unknown_action_type(tmp_path):
    with pytest.raises(ValueError):
        _write_and_run(tmp_path, {"n_disks": 1, "turns": [{"player": "A", "action": "fly", "pole": 1}]})


def test_cli_reports_error_on_missing_player_key(tmp_path):
    with pytest.raises(ValueError):
        _write_and_run(tmp_path, {"n_disks": 1, "turns": [{"action": "skip"}]})


def test_cli_reports_error_on_invalid_player_value(tmp_path):
    with pytest.raises(ValueError):
        _write_and_run(tmp_path, {"n_disks": 1, "turns": [{"player": "Z", "action": "skip"}]})


def test_cli_reports_error_on_missing_pole_for_lift(tmp_path):
    with pytest.raises(ValueError):
        _write_and_run(tmp_path, {"n_disks": 1, "turns": [{"player": "A", "action": "lift"}]})


def test_cli_reports_error_on_unwritable_output_path(tmp_path, capsys):
    """Writing to a path inside a non-existent directory must be reported
    as a clean CLI error, not an uncaught traceback."""
    bad_output = tmp_path / "no-such-dir" / "out.json"
    exit_code = main([str(FIXTURE), "--output", str(bad_output)])
    assert exit_code == 1
    assert "error" in capsys.readouterr().err.lower()


def test_replay_zero_disks_end_to_end():
    """A 0-disk game has nothing to move; it must run cleanly and never win."""
    state, log = run_replay(0, [parse_turn({"player": "A", "action": "skip"})])
    assert state.winners == frozenset()
    assert state.players["A"].pole3 == []
    assert len(log) == 1
    assert log[0]["legal"] is True

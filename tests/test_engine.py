"""Tests that exercise the core engine directly (no CLI, no subprocess)."""

import pytest

from hanoi_crossing.engine import (
    Action,
    ActionType,
    GameState,
    PlayerState,
    apply_action,
    create_initial_state,
    legal_actions,
    legal_mask,
    observe,
)

LIFT1 = Action(ActionType.LIFT, 1)
LIFT2 = Action(ActionType.LIFT, 2)
LIFT3 = Action(ActionType.LIFT, 3)
PLACE1 = Action(ActionType.PLACE, 1)
PLACE2 = Action(ActionType.PLACE, 2)
PLACE3 = Action(ActionType.PLACE, 3)
SKIP = Action(ActionType.SKIP)


def test_initial_deal_odds_to_a_evens_to_b_largest_at_bottom():
    state = create_initial_state(3)
    assert state.players["A"].pole1 == [5, 3, 1]
    assert state.players["B"].pole1 == [6, 4, 2]
    assert state.players["A"].pole3 == []
    assert state.players["B"].pole3 == []
    assert state.shared == []
    assert state.players["A"].hand is None
    assert state.players["B"].hand is None
    assert state.winners == frozenset()
    assert state.move_count == 0


def test_initial_deal_zero_disks():
    state = create_initial_state(0)
    assert state.players["A"].pole1 == []
    assert state.players["B"].pole1 == []


def test_create_initial_state_rejects_negative_n_disks():
    with pytest.raises(ValueError):
        create_initial_state(-1)


def test_action_construction_validates_pole():
    with pytest.raises(ValueError):
        Action(ActionType.LIFT, pole=None)
    with pytest.raises(ValueError):
        Action(ActionType.PLACE, pole=4)
    with pytest.raises(ValueError):
        Action(ActionType.SKIP, pole=1)


def test_action_construction_rejects_bool_pole():
    """bool is a subclass of int in Python, so True == 1 / False == 0;
    without an explicit isinstance check, Action(LIFT, pole=True) would
    silently be accepted as "lift from pole 1"."""
    with pytest.raises(ValueError):
        Action(ActionType.LIFT, pole=True)
    with pytest.raises(ValueError):
        Action(ActionType.PLACE, pole=False)


def test_lift_from_own_pole_is_legal_and_moves_disk_to_hand():
    state = create_initial_state(1)
    result = apply_action(state, "A", LIFT1)
    assert result.legal
    assert result.state.players["A"].hand == 1
    assert result.state.players["A"].pole1 == []
    # original state must be untouched
    assert state.players["A"].hand is None
    assert state.players["A"].pole1 == [1]


def test_lift_from_empty_pole_is_illegal_and_state_unchanged():
    state = create_initial_state(1)
    result = apply_action(state, "A", LIFT3)  # A's pole3 starts empty
    assert not result.legal
    assert "empty" in result.reason
    assert result.state.players["A"].pole3 == []
    assert result.state.players["A"].hand is None
    assert result.state.winners == frozenset()


def test_lift_while_already_holding_is_illegal():
    state = create_initial_state(2)
    lifted = apply_action(state, "A", LIFT1).state
    result = apply_action(lifted, "A", LIFT1)
    assert not result.legal
    assert "already holding" in result.reason
    # state unchanged beyond move_count
    assert result.state.players["A"].hand == lifted.players["A"].hand
    assert result.state.players["A"].pole1 == lifted.players["A"].pole1


def test_place_onto_empty_pole_is_legal():
    state = create_initial_state(1)
    state = apply_action(state, "A", LIFT1).state
    result = apply_action(state, "A", PLACE3)
    assert result.legal
    assert result.state.players["A"].pole3 == [1]
    assert result.state.players["A"].hand is None


def test_place_onto_smaller_top_disk_is_illegal():
    # A holds disk 1 (small); target pole top is disk 1 is not possible with same
    # player, so use B's larger disk 2 stacked under nothing, then a smaller disk
    # attempting to sit under an even smaller one to force "not strictly larger".
    state = create_initial_state(2)  # A: [3,1], B: [4,2]
    state = apply_action(state, "A", LIFT1).state  # A holds 1, A.pole1 = [3]
    state = apply_action(state, "A", PLACE2).state  # shared = [1]
    state = apply_action(state, "B", LIFT1).state  # B holds 2, B.pole1 = [4]
    result = apply_action(state, "B", PLACE2)  # try to stack 2 on top of smaller 1
    assert not result.legal
    assert "not strictly larger" in result.reason
    assert result.state.shared == [1]
    assert result.state.players["B"].hand == 2


def test_place_with_empty_hand_is_illegal():
    state = create_initial_state(1)
    result = apply_action(state, "A", PLACE3)
    assert not result.legal
    assert "hand is empty" in result.reason


def test_skip_is_always_legal_and_never_changes_state():
    state = create_initial_state(2)
    result = apply_action(state, "A", SKIP)
    assert result.legal
    assert result.state.players == state.players
    assert result.state.shared == state.shared
    assert result.state.move_count == state.move_count + 1


def test_cross_player_placement_on_shared_pole():
    """B's disk 2 goes to the shared pole; A legally stacks disk 1 on top of it."""
    state = create_initial_state(1)
    state = apply_action(state, "B", LIFT1).state
    state = apply_action(state, "B", PLACE2).state
    assert state.shared == [2]

    state = apply_action(state, "A", LIFT1).state
    result = apply_action(state, "A", PLACE2)
    assert result.legal
    assert result.state.shared == [2, 1]


def test_either_player_may_lift_from_shared_pole():
    state = create_initial_state(1)
    state = apply_action(state, "A", LIFT1).state
    state = apply_action(state, "A", PLACE2).state  # shared = [1], placed by A
    result = apply_action(state, "B", LIFT2)  # B lifts A's disk from the shared pole
    assert result.legal
    assert result.state.players["B"].hand == 1
    assert result.state.shared == []


def test_readme_worked_example_n1_a_wins():
    """Reproduces the problem statement's N=1 example move-for-move."""
    state = create_initial_state(1)
    r1 = apply_action(state, "A", LIFT1)
    assert r1.legal and r1.state.players["A"].hand == 1
    r2 = apply_action(r1.state, "B", LIFT1)
    assert r2.legal and r2.state.players["B"].hand == 2
    r3 = apply_action(r2.state, "A", PLACE3)
    assert r3.legal
    assert r3.winners == frozenset({"A"})
    assert r3.state.winners == frozenset({"A"})
    assert r3.state.players["A"].pole3 == [1]


def test_game_continues_after_first_win_and_second_player_can_also_win():
    """The engine never halts on a win; winners accumulate across turns."""
    state = create_initial_state(1)
    state = apply_action(state, "A", LIFT1).state
    state = apply_action(state, "B", LIFT1).state
    result_a_wins = apply_action(state, "A", PLACE3)
    assert result_a_wins.winners == frozenset({"A"})
    state = result_a_wins.state

    result_b_wins = apply_action(state, "B", PLACE3)
    assert result_b_wins.winners == frozenset({"B"})
    assert result_b_wins.state.winners == frozenset({"A", "B"})


def test_winners_remain_sticky_after_player_moves_disk_back_off_pole3():
    """Once recorded, a winner stays in state.winners even if they later
    lift their disk back off pole3, un-satisfying the win predicate."""
    state = create_initial_state(1)
    state = apply_action(state, "A", LIFT1).state
    state = apply_action(state, "B", LIFT1).state
    result_a_wins = apply_action(state, "A", PLACE3)
    assert result_a_wins.winners == frozenset({"A"})
    state = result_a_wins.state

    # A undoes their own win by lifting the disk back off pole3.
    undo_result = apply_action(state, "A", LIFT3)
    assert undo_result.legal
    assert undo_result.state.players["A"].pole3 == []
    assert undo_result.state.players["A"].hand == 1
    # winners is sticky: A is still recorded as a winner.
    assert undo_result.state.winners == frozenset({"A"})
    assert undo_result.winners == frozenset()  # no *newly* satisfied winner this call


def test_single_action_can_trigger_at_most_one_new_winner():
    """Every action touches exactly one of {shared, own pole+hand}; since a
    winning PLACE never touches the shared pole and a winning LIFT(2) always
    fills the mover's own hand, one action can newly satisfy at most one
    player's win condition through legitimate play."""
    state = create_initial_state(1)
    state = apply_action(state, "A", LIFT1).state
    state = apply_action(state, "B", LIFT1).state
    result = apply_action(state, "A", PLACE3)
    assert len(result.winners) <= 1


def test_win_check_unions_multiple_newly_satisfied_players_defensively():
    """Whitebox check on the win-detection plumbing itself: apply_action must
    report *every* newly-satisfied player in one call, even from a
    hand-constructed state that would not arise from legitimate play."""
    state = GameState(n_disks=1)
    state.players["A"] = PlayerState(pole1=[], pole3=[1], hand=None)
    state.players["B"] = PlayerState(pole1=[], pole3=[2], hand=None)
    state.shared = []
    result = apply_action(state, "A", SKIP)
    assert result.winners == frozenset({"A", "B"})


def test_observe_hides_opponent_hand_and_private_poles():
    state = GameState(n_disks=1)
    state.players["A"] = PlayerState(pole1=[1], pole3=[], hand=None)
    state.players["B"] = PlayerState(pole1=[], pole3=[2], hand=4)
    state.shared = [7]

    obs_a = observe(state, "A")
    assert obs_a == {"player": "A", "pole1": [1], "pole2": [7], "pole3": [], "hand": None}
    # nothing in A's observation reveals B's hand (4) or B's pole3 ([2])
    assert 4 not in obs_a.values()
    assert [2] not in obs_a.values()


def test_legal_actions_and_legal_mask_are_consistent():
    from hanoi_crossing.engine import ALL_ACTIONS

    state = create_initial_state(2)
    actions = legal_actions(state, "A")
    mask = legal_mask(state, "A")
    assert [a for a, ok in zip(ALL_ACTIONS, mask) if ok] == actions
    assert SKIP in actions  # skip is always legal


def test_legal_actions_never_empty():
    state = create_initial_state(0)  # no disks at all
    assert legal_actions(state, "A") == [SKIP]
    assert legal_actions(state, "B") == [SKIP]


def test_apply_action_does_not_mutate_input_state():
    state = create_initial_state(2)
    snapshot_pole1 = list(state.players["A"].pole1)
    apply_action(state, "A", LIFT1)
    assert state.players["A"].pole1 == snapshot_pole1
    assert state.players["A"].hand is None


def test_engine_is_agnostic_to_turn_order_pattern():
    """A non-alternating pattern (A acting repeatedly) must work fine —
    the engine has no notion of whose turn it "should" be."""
    state = create_initial_state(1)
    for player, action in [("A", LIFT1), ("A", SKIP), ("A", SKIP), ("B", LIFT1)]:
        state = apply_action(state, player, action).state
    assert state.players["A"].hand == 1
    assert state.players["B"].hand == 2

    result = apply_action(state, "A", PLACE3)
    assert result.winners == frozenset({"A"})


def test_unknown_player_raises():
    state = create_initial_state(1)
    with pytest.raises(ValueError):
        apply_action(state, "C", SKIP)
    with pytest.raises(ValueError):
        observe(state, "C")

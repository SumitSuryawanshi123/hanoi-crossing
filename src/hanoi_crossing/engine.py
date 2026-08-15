"""Core Hanoi Crossing engine.

A pure, stateless rules engine for the two-player "Hanoi Crossing" game
(see the project README for the full rules and design rationale). Every
function here takes an explicit ``GameState`` and ``PlayerId`` and returns
a *new* state — nothing is mutated in place and nothing is stored at module
scope. That makes the engine directly reusable as:

  * the environment core of an RL training loop (the loop owns the state
    and decides which agent acts next — the engine has no opinion), or
  * the simulation core of a service holding many concurrent games (each
    game is just a ``GameState`` value; there is no shared mutable state
    to synchronize between games).

The engine intentionally has **no concept of turn order**. Every entry
point takes the acting player explicitly; callers (CLIs, an RL loop, a
service) decide who acts on each step and in what order.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

PlayerId = Literal["A", "B"]
PLAYERS: tuple[PlayerId, PlayerId] = ("A", "B")


class ActionType(str, Enum):
    """The three action kinds a player may take on their turn."""

    LIFT = "lift"
    PLACE = "place"
    SKIP = "skip"


@dataclass(frozen=True)
class Action:
    """A single action, as chosen by a player.

    ``pole`` is always relative to the *acting* player: ``1`` and ``3`` are
    that player's own poles, ``2`` is always the shared pole. There is no
    way to construct an action that references an opponent's pole 1 or 3 —
    visibility is enforced structurally by this addressing scheme, not by
    a legality check.
    """

    type: ActionType
    pole: int | None = None

    def __post_init__(self) -> None:
        if self.type in (ActionType.LIFT, ActionType.PLACE):
            if isinstance(self.pole, bool) or self.pole not in (1, 2, 3):
                raise ValueError(
                    f"{self.type.value} requires pole in {{1, 2, 3}}, got {self.pole!r}"
                )
        elif self.pole is not None:
            raise ValueError("a skip action must not specify a pole")


#: The fixed, state-independent discrete action space (3 lifts + 3 places
#: + skip = 7 actions). Exposed so RL-style callers can build a fixed-size
#: policy output and mask it with :func:`legal_mask`.
ALL_ACTIONS: tuple[Action, ...] = (
    Action(ActionType.LIFT, 1),
    Action(ActionType.LIFT, 2),
    Action(ActionType.LIFT, 3),
    Action(ActionType.PLACE, 1),
    Action(ActionType.PLACE, 2),
    Action(ActionType.PLACE, 3),
    Action(ActionType.SKIP),
)


@dataclass
class PlayerState:
    """One player's private poles (1 and 3) and hand.

    Stacks are stored bottom-to-top, i.e. ``pole1[0]`` is the bottom disk
    and ``pole1[-1]`` is the top (movable) disk.
    """

    pole1: list[int] = field(default_factory=list)
    pole3: list[int] = field(default_factory=list)
    hand: int | None = None

    def copy(self) -> "PlayerState":
        return PlayerState(pole1=list(self.pole1), pole3=list(self.pole3), hand=self.hand)


@dataclass
class GameState:
    """The full, God's-eye game state.

    ``winners`` is *sticky*: once a player satisfies the win condition it
    stays recorded even if disks are later moved again (see README for
    rationale). The engine never forcibly halts on a win — that is a
    frontend/driver policy — so state transitions remain well-defined even
    past the point a winner exists.
    """

    n_disks: int
    shared: list[int] = field(default_factory=list)
    players: dict[PlayerId, PlayerState] = field(default_factory=dict)
    winners: frozenset[PlayerId] = frozenset()
    move_count: int = 0

    def copy(self) -> "GameState":
        return GameState(
            n_disks=self.n_disks,
            shared=list(self.shared),
            players={p: s.copy() for p, s in self.players.items()},
            winners=self.winners,
            move_count=self.move_count,
        )


def create_initial_state(n_disks: int) -> GameState:
    """Deal a fresh game: A gets odds (1, 3, 5, ...), B gets evens (2, 4, ...).

    Both stacks are built largest-at-bottom, as required by the rules.
    """
    if n_disks < 0:
        raise ValueError("n_disks must be a non-negative integer")
    state = GameState(n_disks=n_disks)
    state.players["A"] = PlayerState(pole1=list(range(2 * n_disks - 1, 0, -2)))
    state.players["B"] = PlayerState(pole1=list(range(2 * n_disks, 0, -2)))
    return state


def _check_player(player: PlayerId) -> None:
    if player not in PLAYERS:
        raise ValueError(f"unknown player {player!r}; expected 'A' or 'B'")


def observe(state: GameState, player: PlayerId) -> dict:
    """The partial observation available to ``player``.

    Contains only that player's own ``pole1``/``pole3``/``hand`` plus the
    shared ``pole2`` — nothing about the opponent's private poles or hand,
    per the rule that neither player can see those. This is deliberately
    the *only* information :func:`legal_actions` and the reference random
    player are allowed to use, so a real external agent (human, scripted,
    or RL policy) never needs more than this to act correctly.
    """
    _check_player(player)
    own = state.players[player]
    return {
        "player": player,
        "pole1": list(own.pole1),
        "pole2": list(state.shared),
        "pole3": list(own.pole3),
        "hand": own.hand,
    }


def _is_legal(obs: dict, action: Action) -> bool:
    if action.type == ActionType.SKIP:
        return True
    pole = obs[f"pole{action.pole}"]
    if action.type == ActionType.LIFT:
        return obs["hand"] is None and len(pole) > 0
    # PLACE
    if obs["hand"] is None:
        return False
    top = pole[-1] if pole else None
    return top is None or top > obs["hand"]


def _illegal_reason(obs: dict, action: Action) -> str:
    if action.type == ActionType.LIFT:
        if obs["hand"] is not None:
            return "cannot lift: hand is already holding a disk"
        return f"cannot lift: pole {action.pole} is empty"
    if action.type == ActionType.PLACE:
        if obs["hand"] is None:
            return "cannot place: hand is empty"
        pole = obs[f"pole{action.pole}"]
        top = pole[-1] if pole else None
        return (
            f"cannot place disk {obs['hand']} onto pole {action.pole}: "
            f"top disk {top} is not strictly larger"
        )
    return "illegal action"  # pragma: no cover - SKIP is always legal


def legal_actions(state: GameState, player: PlayerId) -> list[Action]:
    """All actions ``player`` may legally take right now.

    Computed purely from :func:`observe`, so legality never depends on
    hidden opponent information. Always non-empty: ``SKIP`` is always
    legal.
    """
    obs = observe(state, player)
    return [a for a in ALL_ACTIONS if _is_legal(obs, a)]


def legal_mask(state: GameState, player: PlayerId) -> list[bool]:
    """Boolean mask aligned with :data:`ALL_ACTIONS`, for RL-style action masking."""
    obs = observe(state, player)
    return [_is_legal(obs, a) for a in ALL_ACTIONS]


@dataclass(frozen=True)
class ActionResult:
    """The outcome of a single :func:`apply_action` call."""

    state: GameState
    legal: bool
    reason: str | None
    #: Players whose win condition newly became true *this call* (may
    #: contain both players — see README on simultaneous double-wins).
    winners: frozenset[PlayerId]


def _pole_ref(state: GameState, player: PlayerId, pole: int) -> list[int]:
    if pole == 2:
        return state.shared
    own = state.players[player]
    return own.pole1 if pole == 1 else own.pole3


def _has_won(state: GameState, player: PlayerId) -> bool:
    """A player wins iff: empty hand, empty own pole1, empty shared pole,
    and a non-empty own pole3 (per the rules' "only pole 3 has disks")."""
    own = state.players[player]
    return own.hand is None and not own.pole1 and not state.shared and bool(own.pole3)


def apply_action(state: GameState, player: PlayerId, action: Action) -> ActionResult:
    """Apply ``action`` on behalf of ``player`` and return the outcome.

    Legal actions produce a new state reflecting the lift/place/skip and
    then re-check *both* players' win condition (clearing the shared pole
    can complete the other player's win too). Illegal actions leave every
    field of the returned state identical to the input except
    ``move_count`` (bumped so callers can budget wasted turns), per the
    rule that an illegal action wastes the turn without changing the game.
    """
    _check_player(player)
    obs = observe(state, player)
    new_state = state.copy()
    new_state.move_count += 1

    if not _is_legal(obs, action):
        return ActionResult(
            state=new_state, legal=False, reason=_illegal_reason(obs, action), winners=frozenset()
        )

    own = new_state.players[player]
    if action.type == ActionType.LIFT:
        pole = _pole_ref(new_state, player, action.pole)
        own.hand = pole.pop()
    elif action.type == ActionType.PLACE:
        pole = _pole_ref(new_state, player, action.pole)
        pole.append(own.hand)
        own.hand = None
    # SKIP: no state change beyond move_count.

    newly_won = frozenset(
        p for p in PLAYERS if p not in state.winners and _has_won(new_state, p)
    )
    new_state.winners = state.winners | newly_won
    return ActionResult(state=new_state, legal=True, reason=None, winners=newly_won)


def has_winner(state: GameState) -> bool:
    """Convenience helper: has anyone won (so far, stickily) in ``state``?"""
    return bool(state.winners)

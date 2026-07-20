"""
Text-based Tower of Hanoi environment and reproducible instance generator.
"""

from __future__ import annotations

import random
import re
from typing import Any

PegState = dict[str, list[int]]
Move = tuple[str, str]


def _sorted_goal_stack(num_disks: int) -> list[int]:
    return list(range(num_disks, 0, -1))


def _copy_state(state: PegState) -> PegState:
    return {peg: list(stack) for peg, stack in state.items()}


def _state_to_tuple(state: PegState) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (tuple(state["A"]), tuple(state["B"]), tuple(state["C"]))


def _legal_moves(state: PegState) -> list[Move]:
    moves: list[Move] = []
    for src in ("A", "B", "C"):
        if not state[src]:
            continue
        disk = state[src][-1]
        for dst in ("A", "B", "C"):
            if src == dst:
                continue
            if not state[dst] or state[dst][-1] > disk:
                moves.append((src, dst))
    return moves


def _apply_move(state: PegState, move: Move) -> PegState:
    src, dst = move
    nxt = _copy_state(state)
    disk = nxt[src].pop()
    nxt[dst].append(disk)
    return nxt


def _is_goal(state: PegState, num_disks: int) -> bool:
    return state["A"] == [] and state["B"] == [] and state["C"] == _sorted_goal_stack(num_disks)


def _shortest_path_to_goal(state: PegState, num_disks: int) -> list[Move]:
    """BFS shortest legal move sequence from any valid state to canonical goal."""
    if _is_goal(state, num_disks):
        return []
    start_key = _state_to_tuple(state)
    queue: list[tuple[PegState, list[Move]]] = [(_copy_state(state), [])]
    seen = {start_key}
    while queue:
        cur, path = queue.pop(0)
        for mv in _legal_moves(cur):
            nxt = _apply_move(cur, mv)
            key = _state_to_tuple(nxt)
            if key in seen:
                continue
            npath = path + [mv]
            if _is_goal(nxt, num_disks):
                return npath
            seen.add(key)
            queue.append((nxt, npath))
    return []


def _format_move(move: Move) -> str:
    return f"{move[0]}->{move[1]}"


def _render_peg(disks: list[int]) -> str:
    return str(disks) if disks else "(empty)"


def _parse_action(action: str) -> Move | None:
    text = action.strip()
    if not text:
        return None
    upper = text.upper()
    # Accept arrows and compact forms.
    upper = upper.replace("→", "->")
    m = re.search(r"\b([ABC])\s*->\s*([ABC])\b", upper)
    if m:
        src, dst = m.group(1), m.group(2)
        return (src, dst) if src != dst else None
    # Accept "Move disk from A to C", "A to C", "move A C", "a c".
    letters = re.findall(r"\b([ABC])\b", upper)
    if len(letters) >= 2:
        src, dst = letters[0], letters[1]
        return (src, dst) if src != dst else None
    return None


class TowerOfHanoiEnv:
    """
    Text environment with reset() and step(action) API used by the agent loop.
    """

    def __init__(
        self,
        task: dict,
        max_steps: int = 50,
        *,
        include_valid_moves: bool = False,
    ) -> None:
        self._task = task
        self._num_disks = int(task["num_disks"])
        self._initial_state: PegState = _copy_state(task["initial_state"])
        self._state: PegState = _copy_state(self._initial_state)
        self._error_message = ""
        self._max_steps = int(max_steps)
        self._include_valid_moves = bool(include_valid_moves)
        self.observation: str = ""
        self.done = False
        self.task_success = False
        self.step_results: list[dict[str, Any]] = []
        self.current_step = 0
        self._optimal_solution: list[Move] = [tuple(m) for m in task.get("optimal_solution", [])]
        if not self._optimal_solution:
            self._optimal_solution = _shortest_path_to_goal(self._state, self._num_disks)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def _render_observation(self) -> str:
        lines = [
            "Current state:",
            f"  Peg A: {_render_peg(self._state['A'])}",
            f"  Peg B: {_render_peg(self._state['B'])}",
            f"  Peg C: {_render_peg(self._state['C'])}",
            "Each peg's disks are listed bottom-to-top, so the last (rightmost) number is the top disk.",
            "Each number is a disk's size; a larger number is a larger disk (1 is the smallest).",
            f"Goal state: Peg C holds all {self._num_disks} disks, Peg A and Peg B are empty.",
            "Rules: move only the top disk from a peg; never put a larger disk on a smaller one.",
            "Reply with a single move: two peg letters, e.g. <source>-><target> or <source> to <target>.",
        ]
        if self._include_valid_moves:
            valid = ", ".join(_format_move(mv) for mv in _legal_moves(self._state))
            reply_line_idx = next(
                i for i, line in enumerate(lines) if line.startswith("Reply with a single move")
            )
            lines.insert(
                reply_line_idx,
                f"Valid moves: {valid} — choose exactly one of these.",
            )
        if self._error_message:
            lines.insert(0, f"Illegal move: {self._error_message}")
        return "\n".join(lines)

    def reset(self) -> str:
        self._state = _copy_state(self._initial_state)
        self._error_message = ""
        self.done = False
        self.task_success = False
        self.current_step = 0
        self.step_results = []
        self._optimal_solution = _shortest_path_to_goal(self._state, self._num_disks)
        self.observation = self._render_observation()
        return self.observation

    def step(self, action: str) -> str:
        if self.done:
            return self.observation

        state_before = _copy_state(self._state)
        parsed = _parse_action(action)
        correctness: str
        self._error_message = ""

        if parsed is None:
            correctness = "illegal"
            self._error_message = "Could not parse action."
            state_after = _copy_state(self._state)
        else:
            src, dst = parsed
            legal = parsed in _legal_moves(self._state)
            if not legal:
                correctness = "illegal"
                self._error_message = "Move violates disk-size rule or source is empty."
                state_after = _copy_state(self._state)
            else:
                dist_before = len(_shortest_path_to_goal(state_before, self._num_disks))
                self._state = _apply_move(self._state, parsed)
                state_after = _copy_state(self._state)
                dist_after = len(_shortest_path_to_goal(state_after, self._num_disks))
                if dist_after == dist_before - 1:
                    correctness = "optimal"
                else:
                    correctness = "legal"
                self._optimal_solution = _shortest_path_to_goal(self._state, self._num_disks)

        self.step_results.append(
            {
                "step_index": self.current_step,
                "action_raw": action,
                "action_parsed": parsed,
                "correctness": correctness,
                "optimal_moves_remaining": len(
                    _shortest_path_to_goal(state_after, self._num_disks)
                ),
                "state_before": state_before,
                "state_after": state_after,
            }
        )
        self.current_step += 1

        if _is_goal(self._state, self._num_disks):
            self.done = True
            self.task_success = True
        elif self.current_step >= self._max_steps:
            self.done = True
            self.task_success = False

        self.observation = self._render_observation()
        return self.observation


def _enumerate_reachable_states(num_disks: int) -> list[PegState]:
    """All 3**num_disks valid states via BFS from the fully-scrambled start (every disk on A).
    Confirms the well-known 3**n state-space size for classic 3-peg Tower of Hanoi: a valid
    state is fully determined by which peg each disk sits on (within-peg order is forced by the
    smaller-on-top rule), so there are no further degrees of freedom.
    """
    start: PegState = {"A": _sorted_goal_stack(num_disks), "B": [], "C": []}
    seen: dict[tuple[tuple[int, ...], ...], PegState] = {_state_to_tuple(start): start}
    frontier = [start]
    while frontier:
        next_frontier = []
        for st in frontier:
            for mv in _legal_moves(st):
                nxt = _apply_move(st, mv)
                key = _state_to_tuple(nxt)
                if key not in seen:
                    seen[key] = nxt
                    next_frontier.append(nxt)
        frontier = next_frontier
    return list(seen.values())


class _StatePool:
    """Per-disk-count pool of distinct reachable states, drawn without replacement so that
    generate_instances(..., partial_start_mode="random_scramble") cannot repeat a state until
    every other reachable state for that disk count has been used once. A short random walk
    (bounded by partial_start_range, as in "optimal_prefix" mode) only explores a small
    neighborhood around the fully-scrambled start -- verified empirically to leave most of the
    3**num_disks state space unvisited even at walk lengths of 50 -- so per-instance random
    walks were replaced with this exhaustive, shuffled-without-replacement draw.
    """

    def __init__(self) -> None:
        self._pools: dict[int, list[PegState]] = {}
        self._cursors: dict[int, int] = {}

    def draw(self, num_disks: int, rng: random.Random) -> PegState:
        if num_disks not in self._pools:
            states = _enumerate_reachable_states(num_disks)
            goal_key = _state_to_tuple({"A": [], "B": [], "C": _sorted_goal_stack(num_disks)})
            states = [s for s in states if _state_to_tuple(s) != goal_key]
            rng.shuffle(states)
            self._pools[num_disks] = states
            self._cursors[num_disks] = 0
        pool = self._pools[num_disks]
        cursor = self._cursors[num_disks]
        if cursor >= len(pool):
            rng.shuffle(pool)
            cursor = 0
        self._cursors[num_disks] = cursor + 1
        return _copy_state(pool[cursor])


def generate_instances(
    n: int,
    seed: int,
    num_disks_range: tuple[int, int] = (3, 4),
    partial_start_range: tuple[int, int] = (0, 3),
    partial_start_mode: str = "optimal_prefix",
) -> list[dict]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if partial_start_mode not in ("optimal_prefix", "random_scramble"):
        raise ValueError("partial_start_mode must be 'optimal_prefix' or 'random_scramble'")
    d_lo, d_hi = int(num_disks_range[0]), int(num_disks_range[1])
    p_lo, p_hi = int(partial_start_range[0]), int(partial_start_range[1])
    if d_lo > d_hi or d_lo < 1:
        raise ValueError("num_disks_range must be [lo, hi] with lo >= 1")
    if p_lo > p_hi or p_lo < 0:
        raise ValueError("partial_start_range must be [lo, hi] with lo >= 0")

    rng = random.Random(seed)
    pool = _StatePool() if partial_start_mode == "random_scramble" else None
    instances: list[dict] = []
    for i in range(n):
        sub = random.Random(rng.randint(0, 2**31 - 1))
        num_disks = sub.randint(d_lo, d_hi)
        if partial_start_mode == "random_scramble":
            assert pool is not None
            state = pool.draw(num_disks, sub)
            applied_partial = -1  # not meaningful in this mode -- drawn from the full state pool
        else:
            requested_partial = sub.randint(p_lo, p_hi)
            state = {"A": _sorted_goal_stack(num_disks), "B": [], "C": []}
            applied_partial = 0
            for _ in range(requested_partial):
                path = _shortest_path_to_goal(state, num_disks)
                if not path:
                    break
                state = _apply_move(state, path[0])
                applied_partial += 1
        optimal_solution = _shortest_path_to_goal(state, num_disks)
        max_steps = max(1, len(optimal_solution) * 3)
        instances.append(
            {
                "id": f"tower_of_hanoi_{i}",
                "num_disks": num_disks,
                "initial_state": _copy_state(state),
                "goal_state": {"A": [], "B": [], "C": _sorted_goal_stack(num_disks)},
                "optimal_solution": list(optimal_solution),
                "optimal_steps": len(optimal_solution),
                "max_steps": max_steps,
                "partial_start_moves": applied_partial,
            }
        )
    return instances

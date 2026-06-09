"""Small teaching utilities for the optimization micro-labs.

The module intentionally keeps the algorithms readable. Pyomo is used only for
explicit LP/MILP modeling; constructive heuristics and metaheuristics are plain
Python so students can inspect every moving part.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence


Record = Mapping[str, Any]
State = Any
Objective = Callable[[State], float]
Neighbor = Callable[[State, random.Random], State]


@dataclass(frozen=True)
class Limits:
    """Resource limits used by the outage task-selection labs."""

    max_duration: float
    max_budget: float
    max_dose: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "Limits":
        return cls(
            max_duration=float(values["max_duration"]),
            max_budget=float(values["max_budget"]),
            max_dose=float(values["max_dose"]),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "max_duration": self.max_duration,
            "max_budget": self.max_budget,
            "max_dose": self.max_dose,
        }


DEFAULT_LIMITS = Limits(max_duration=140.0, max_budget=850.0, max_dose=380.0)


class KnapsackProblem:
    """Minimal 0-1 knapsack problem kept for the introductory demos."""

    def __init__(self, values: list[int], weights: list[int], capacity: int):
        self.values = values
        self.weights = weights
        self.capacity = capacity
        self.num_items = len(values)

    def evaluate(self, state: list[int]) -> int:
        total_weight = sum(w * s for w, s in zip(self.weights, state))
        if total_weight > self.capacity:
            return 0
        return sum(v * s for v, s in zip(self.values, state))

    def get_random_state(self, rng: random.Random | None = None) -> list[int]:
        rng = rng or random
        return [rng.choice([0, 1]) for _ in range(self.num_items)]

    def get_neighbor(self, state: list[int], rng: random.Random | None = None) -> list[int]:
        rng = rng or random
        neighbor = state.copy()
        idx = rng.randint(0, self.num_items - 1)
        neighbor[idx] = 1 - neighbor[idx]
        return neighbor


def generate_outage_case(
    seed: int = 42,
    n_tasks: int = 40,
    n_zones: int = 12,
) -> list[dict[str, Any]]:
    """Generate synthetic outage work packages.

    The fields are intentionally plausible rather than realistic. They are safe
    teaching data: no real plant parameters and no operational detail.
    """

    rng = random.Random(seed)
    crew_types = ["mechanical", "electrical", "i_and_c", "radiation", "fuel"]
    safety_groups = ["A", "B", "C", "none"]
    zone_dose = {f"Z{z:02d}": rng.uniform(0.4, 3.6) for z in range(1, n_zones + 1)}

    tasks: list[dict[str, Any]] = []
    for idx in range(1, n_tasks + 1):
        zone = f"Z{rng.randint(1, n_zones):02d}"
        duration = rng.randint(3, 18)
        crew_type = rng.choice(crew_types)
        work_cost = rng.randint(20, 140)
        delay_cost = rng.randint(15, 120)
        dose_rate = round(zone_dose[zone] * rng.uniform(0.75, 1.35), 2)
        value = rng.randint(30, 180) + int(0.45 * delay_cost)

        predecessors: list[str] = []
        if idx > 3 and rng.random() < 0.28:
            pred_count = 1 if rng.random() < 0.85 else 2
            predecessors = [
                f"T{rng.randint(1, idx - 1):03d}" for _ in range(pred_count)
            ]
            predecessors = sorted(set(predecessors))

        tasks.append(
            {
                "task_id": f"T{idx:03d}",
                "duration": duration,
                "crew_type": crew_type,
                "zone": zone,
                "value": value,
                "work_cost": work_cost,
                "delay_cost": delay_cost,
                "dose_rate": dose_rate,
                "safety_group": rng.choice(safety_groups),
                "predecessors": predecessors,
            }
        )
    return tasks


def outage_task_dose(task: Record) -> float:
    return float(task["duration"]) * float(task["dose_rate"])


def evaluate_selection(
    tasks: Sequence[Record],
    selected_task_ids: Iterable[str],
    limits: Limits | Mapping[str, float] = DEFAULT_LIMITS,
) -> dict[str, Any]:
    """Evaluate resource use and objective value for a selected task set."""

    if not isinstance(limits, Limits):
        limits = Limits.from_mapping(limits)

    selected = set(selected_task_ids)
    rows = [task for task in tasks if task["task_id"] in selected]
    total_duration = sum(float(task["duration"]) for task in rows)
    total_budget = sum(float(task["work_cost"]) for task in rows)
    total_dose = sum(outage_task_dose(task) for task in rows)
    total_value = sum(float(task["value"]) for task in rows)

    return {
        "selected_task_ids": [task["task_id"] for task in rows],
        "n_tasks": len(rows),
        "total_duration": total_duration,
        "total_budget": total_budget,
        "total_dose": total_dose,
        "total_value": total_value,
        "feasible": (
            total_duration <= limits.max_duration
            and total_budget <= limits.max_budget
            and total_dose <= limits.max_dose
        ),
    }


def greedy_select(
    tasks: Sequence[Record],
    limits: Limits | Mapping[str, float] = DEFAULT_LIMITS,
    priority: Callable[[Record], float] | None = None,
) -> dict[str, Any]:
    """Select outage tasks greedily by a priority score.

    Default priority is value density per unit of duration. The function returns
    both the selected IDs and aggregate resource usage.
    """

    if not isinstance(limits, Limits):
        limits = Limits.from_mapping(limits)
    priority = priority or (lambda task: float(task["value"]) / float(task["duration"]))

    selected: list[str] = []
    used_duration = 0.0
    used_budget = 0.0
    used_dose = 0.0

    for task in sorted(tasks, key=priority, reverse=True):
        next_duration = used_duration + float(task["duration"])
        next_budget = used_budget + float(task["work_cost"])
        next_dose = used_dose + outage_task_dose(task)

        if (
            next_duration <= limits.max_duration
            and next_budget <= limits.max_budget
            and next_dose <= limits.max_dose
        ):
            selected.append(str(task["task_id"]))
            used_duration = next_duration
            used_budget = next_budget
            used_dose = next_dose

    result = evaluate_selection(tasks, selected, limits)
    result["method"] = "greedy"
    return result


def build_pyomo_outage_model(
    tasks: Sequence[Record],
    limits: Limits | Mapping[str, float] = DEFAULT_LIMITS,
    binary: bool = True,
):
    """Build a Pyomo task-selection model.

    The same model becomes an LP relaxation with ``binary=False`` and a MILP
    with ``binary=True``.
    """

    try:
        import pyomo.environ as pyo
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by users
        raise RuntimeError(
            "Pyomo is required for this lab. Install project dependencies first."
        ) from exc

    if not isinstance(limits, Limits):
        limits = Limits.from_mapping(limits)

    rows = list(tasks)
    model = pyo.ConcreteModel(name="outage_task_selection")
    model.I = pyo.RangeSet(0, len(rows) - 1)
    model.x = pyo.Var(model.I, domain=pyo.Binary if binary else pyo.UnitInterval)

    model.objective = pyo.Objective(
        expr=sum(float(rows[i]["value"]) * model.x[i] for i in model.I),
        sense=pyo.maximize,
    )
    model.duration_limit = pyo.Constraint(
        expr=sum(float(rows[i]["duration"]) * model.x[i] for i in model.I)
        <= limits.max_duration
    )
    model.budget_limit = pyo.Constraint(
        expr=sum(float(rows[i]["work_cost"]) * model.x[i] for i in model.I)
        <= limits.max_budget
    )
    model.dose_limit = pyo.Constraint(
        expr=sum(outage_task_dose(rows[i]) * model.x[i] for i in model.I)
        <= limits.max_dose
    )

    model._task_rows = rows
    model._limits = limits
    model._binary = binary
    return model


def solve_pyomo_model(model, solver: str = "appsi_highs") -> dict[str, Any]:
    """Solve a Pyomo model and return a small teaching-friendly summary."""

    try:
        import pyomo.environ as pyo
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by users
        raise RuntimeError(
            "Pyomo is required for this lab. Install project dependencies first."
        ) from exc

    opt = pyo.SolverFactory(solver)
    if opt is None or not opt.available(exception_flag=False):
        raise RuntimeError(
            f"Solver '{solver}' is not available. Install highspy or choose another solver."
        )

    results = opt.solve(model)
    rows = list(model._task_rows)
    values = [float(pyo.value(model.x[i])) for i in model.I]
    selected = [rows[i]["task_id"] for i, value in enumerate(values) if value > 1e-6]
    evaluation = evaluate_selection(rows, selected, model._limits)

    return {
        "success": _is_successful_termination(results),
        "solver": solver,
        "termination_condition": _termination_condition(results),
        "objective": float(pyo.value(model.objective)),
        "variables": [
            {"task_id": rows[i]["task_id"], "x": values[i]} for i in range(len(rows))
        ],
        "selected_task_ids": selected,
        "evaluation": evaluation,
        "raw_results": results,
    }


def _termination_condition(results: Any) -> str:
    solver = getattr(results, "solver", None)
    condition = getattr(solver, "termination_condition", None)
    if condition is not None:
        return str(condition)
    termination = getattr(results, "termination_condition", None)
    return str(termination)


def _is_successful_termination(results: Any) -> bool:
    return _termination_condition(results).lower() in {
        "optimal",
        "locallyoptimal",
        "feasible",
    }


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(a, b)


def route_cost(route: Sequence[Any], distance: Callable[[Any, Any], float]) -> float:
    """Cycle cost: visit every node and return to the start."""

    if len(route) < 2:
        return 0.0
    return sum(
        distance(route[i], route[(i + 1) % len(route)]) for i in range(len(route))
    )


def nearest_neighbor_route(
    points: Mapping[Any, Sequence[float]] | Sequence[Sequence[float]],
    start: Any | None = None,
) -> list[Any]:
    """Build a TSP route by repeatedly visiting the nearest unvisited point."""

    if isinstance(points, Mapping):
        ids = list(points.keys())
        coords = points
    else:
        ids = list(range(len(points)))
        coords = {idx: points[idx] for idx in ids}

    if not ids:
        return []
    current = start if start is not None else ids[0]
    if current not in ids:
        raise ValueError("start must be one of the point IDs")

    unvisited = set(ids)
    route = [current]
    unvisited.remove(current)

    while unvisited:
        current = min(
            unvisited,
            key=lambda candidate: euclidean_distance(coords[route[-1]], coords[candidate]),
        )
        route.append(current)
        unvisited.remove(current)

    return route


def two_opt(
    route: Sequence[Any],
    distance: Callable[[Any, Any], float],
    max_iterations: int = 100,
) -> tuple[list[Any], float, list[float]]:
    """Classic 2-opt local search for a closed TSP route."""

    best_route = list(route)
    best_cost = route_cost(best_route, distance)
    history = [best_cost]

    for _ in range(max_iterations):
        improved = False
        for i in range(1, len(best_route) - 2):
            for k in range(i + 1, len(best_route)):
                if k - i == 1:
                    continue
                candidate = best_route[:i] + best_route[i:k][::-1] + best_route[k:]
                candidate_cost = route_cost(candidate, distance)
                if candidate_cost + 1e-12 < best_cost:
                    best_route = candidate
                    best_cost = candidate_cost
                    history.append(best_cost)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best_route, best_cost, history


def tsp_segment_neighbor(route: Sequence[Any], rng: random.Random) -> list[Any]:
    """Random 2-opt-like neighbor for simulated annealing."""

    candidate = list(route)
    if len(candidate) < 4:
        return candidate
    i, k = sorted(rng.sample(range(1, len(candidate)), 2))
    candidate[i:k] = reversed(candidate[i:k])
    return candidate


def simulated_annealing(
    initial_state: State,
    objective: Objective,
    neighbor: Neighbor,
    seed: int = 42,
    init_temp: float = 100.0,
    cooling_rate: float = 0.97,
    max_iters: int = 1000,
    minimize: bool = True,
) -> dict[str, Any]:
    """Generic simulated annealing implementation in clean Python."""

    rng = random.Random(seed)
    current_state = _copy_state(initial_state)
    current_score = float(objective(current_state))
    best_state = _copy_state(current_state)
    best_score = current_score
    temperature = init_temp
    accepted_worse = 0
    history: list[dict[str, float | bool]] = []

    for iteration in range(max_iters):
        candidate_state = neighbor(_copy_state(current_state), rng)
        candidate_score = float(objective(candidate_state))
        delta = candidate_score - current_score

        if minimize:
            improves = delta <= 0
            probability = math.exp(-delta / temperature) if temperature > 1e-12 else 0.0
        else:
            improves = delta >= 0
            probability = math.exp(delta / temperature) if temperature > 1e-12 else 0.0

        accepted = improves or rng.random() < probability
        if accepted:
            if not improves:
                accepted_worse += 1
            current_state = candidate_state
            current_score = candidate_score

        better_than_best = current_score < best_score if minimize else current_score > best_score
        if better_than_best:
            best_state = _copy_state(current_state)
            best_score = current_score

        history.append(
            {
                "iteration": iteration,
                "temperature": temperature,
                "current_score": current_score,
                "best_score": best_score,
                "accepted": accepted,
                "accepted_worse": accepted and not improves,
            }
        )
        temperature *= cooling_rate

    return {
        "best_state": best_state,
        "best_score": best_score,
        "history": history,
        "accepted_worse": accepted_worse,
        "accepted_worse_rate": accepted_worse / max(1, max_iters),
    }


def _copy_state(state: State) -> State:
    return state.copy() if hasattr(state, "copy") else state


def knapsack_local_search(
    problem: KnapsackProblem,
    max_iters: int = 1000,
    seed: int = 42,
) -> tuple[list[int], int]:
    """First-choice hill climbing for the introductory knapsack demo."""

    rng = random.Random(seed)
    current_state = problem.get_random_state(rng)
    current_value = problem.evaluate(current_state)

    for _ in range(max_iters):
        neighbor_state = problem.get_neighbor(current_state, rng)
        neighbor_value = problem.evaluate(neighbor_state)
        if neighbor_value > current_value:
            current_state = neighbor_state
            current_value = neighbor_value

    return current_state, current_value


def knapsack_simulated_annealing(
    problem: KnapsackProblem,
    init_temp: float = 100.0,
    cooling_rate: float = 0.95,
    max_iters: int = 1000,
    seed: int = 42,
) -> tuple[list[int], int]:
    """Compatibility wrapper for the earlier knapsack-only SA demo."""

    initial_state = problem.get_random_state(random.Random(seed))
    result = simulated_annealing(
        initial_state=initial_state,
        objective=lambda state: problem.evaluate(state),
        neighbor=lambda state, rng: problem.get_neighbor(state, rng),
        seed=seed,
        init_temp=init_temp,
        cooling_rate=cooling_rate,
        max_iters=max_iters,
        minimize=False,
    )
    return result["best_state"], int(result["best_score"])


def pareto_front(
    records: Sequence[Record],
    objectives: Sequence[str | tuple[str, str]],
) -> list[dict[str, Any]]:
    """Return nondominated records for minimization/maximization objectives."""

    specs = [
        (objective, "min") if isinstance(objective, str) else objective
        for objective in objectives
    ]

    front: list[dict[str, Any]] = []
    for candidate in records:
        if not any(_dominates(other, candidate, specs) for other in records if other is not candidate):
            front.append(dict(candidate))
    return front


def _dominates(
    left: Record,
    right: Record,
    specs: Sequence[tuple[str, str]],
) -> bool:
    no_worse = True
    strictly_better = False

    for key, direction in specs:
        left_value = float(left[key])
        right_value = float(right[key])
        if direction == "min":
            if left_value > right_value:
                no_worse = False
                break
            if left_value < right_value:
                strictly_better = True
        elif direction == "max":
            if left_value < right_value:
                no_worse = False
                break
            if left_value > right_value:
                strictly_better = True
        else:
            raise ValueError("Objective direction must be 'min' or 'max'")

    return no_worse and strictly_better


def make_greedy_counterexample() -> tuple[list[dict[str, Any]], Limits]:
    """Small task set where value-density greedy loses to the MILP optimum."""

    tasks = [
        {
            "task_id": "A",
            "duration": 6,
            "crew_type": "mechanical",
            "zone": "Z01",
            "value": 10,
            "work_cost": 1,
            "delay_cost": 10,
            "dose_rate": 1.0,
            "safety_group": "none",
            "predecessors": [],
        },
        {
            "task_id": "B",
            "duration": 5,
            "crew_type": "mechanical",
            "zone": "Z01",
            "value": 7,
            "work_cost": 1,
            "delay_cost": 7,
            "dose_rate": 1.0,
            "safety_group": "none",
            "predecessors": [],
        },
        {
            "task_id": "C",
            "duration": 5,
            "crew_type": "mechanical",
            "zone": "Z01",
            "value": 7,
            "work_cost": 1,
            "delay_cost": 7,
            "dose_rate": 1.0,
            "safety_group": "none",
            "predecessors": [],
        },
    ]
    limits = Limits(max_duration=10, max_budget=10, max_dose=10)
    return tasks, limits


def local_search(problem: KnapsackProblem, max_iters: int = 1000) -> tuple[list[int], int]:
    """Backward-compatible alias for the original knapsack local search demo."""

    return knapsack_local_search(problem, max_iters=max_iters)


if __name__ == "__main__":
    values = [10, 20, 15, 30, 40, 12, 18, 5, 25, 35]
    weights = [5, 12, 8, 17, 22, 6, 9, 3, 11, 15]
    knapsack = KnapsackProblem(values=values, weights=weights, capacity=50)

    ls_state, ls_val = knapsack_local_search(knapsack, max_iters=500, seed=42)
    sa_state, sa_val = knapsack_simulated_annealing(
        knapsack, init_temp=50.0, cooling_rate=0.98, max_iters=500, seed=42
    )

    print("Local Search:", ls_val, ls_state)
    print("Simulated Annealing:", sa_val, sa_state)

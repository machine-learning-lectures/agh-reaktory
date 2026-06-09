from __future__ import annotations

import math

import pytest

from utilis.library import (
    DEFAULT_LIMITS,
    build_pyomo_outage_model,
    evaluate_selection,
    euclidean_distance,
    generate_outage_case,
    greedy_select,
    make_greedy_counterexample,
    nearest_neighbor_route,
    pareto_front,
    route_cost,
    simulated_annealing,
    solve_pyomo_model,
    tsp_segment_neighbor,
    two_opt,
)


def test_generate_outage_case_has_required_fields_and_is_deterministic():
    left = generate_outage_case(seed=7, n_tasks=6, n_zones=3)
    right = generate_outage_case(seed=7, n_tasks=6, n_zones=3)

    assert left == right
    assert len(left) == 6
    assert {
        "task_id",
        "duration",
        "crew_type",
        "zone",
        "value",
        "work_cost",
        "delay_cost",
        "dose_rate",
        "safety_group",
        "predecessors",
    }.issubset(left[0])


def test_greedy_returns_feasible_selection():
    tasks = generate_outage_case(seed=11, n_tasks=20, n_zones=5)
    result = greedy_select(tasks, DEFAULT_LIMITS)
    evaluation = evaluate_selection(tasks, result["selected_task_ids"], DEFAULT_LIMITS)

    assert result["feasible"] is True
    assert evaluation["feasible"] is True
    assert result["total_value"] == evaluation["total_value"]


def test_greedy_counterexample_is_worse_than_milp_if_solver_available():
    pyo = pytest.importorskip("pyomo.environ")
    if not pyo.SolverFactory("appsi_highs").available(exception_flag=False):
        pytest.skip("appsi_highs solver is not available")

    tasks, limits = make_greedy_counterexample()
    greedy = greedy_select(tasks, limits)
    model = build_pyomo_outage_model(tasks, limits, binary=True)
    milp = solve_pyomo_model(model)

    assert greedy["total_value"] == 10
    assert milp["objective"] == pytest.approx(14)
    assert set(milp["selected_task_ids"]) == {"B", "C"}
    assert milp["objective"] > greedy["total_value"]


def test_pyomo_lp_relaxation_can_return_fractional_solution_if_solver_available():
    pyo = pytest.importorskip("pyomo.environ")
    if not pyo.SolverFactory("appsi_highs").available(exception_flag=False):
        pytest.skip("appsi_highs solver is not available")

    tasks, limits = make_greedy_counterexample()
    model = build_pyomo_outage_model(tasks, limits, binary=False)
    result = solve_pyomo_model(model)
    xs = [item["x"] for item in result["variables"]]

    assert result["success"] is True
    assert any(0.0 < x < 1.0 for x in xs)
    assert result["objective"] >= 14


def test_two_opt_never_makes_route_worse():
    points = {
        "A": (0.0, 0.0),
        "B": (0.0, 1.0),
        "C": (1.0, 1.0),
        "D": (1.0, 0.0),
        "E": (0.45, 0.5),
    }
    route = ["A", "C", "B", "D", "E"]
    distance = lambda a, b: euclidean_distance(points[a], points[b])

    initial = route_cost(route, distance)
    improved_route, improved_cost, history = two_opt(route, distance)

    assert set(improved_route) == set(route)
    assert improved_cost <= initial
    assert history[0] == pytest.approx(initial)
    assert history[-1] == pytest.approx(improved_cost)


def test_nearest_neighbor_visits_each_point_once():
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    route = nearest_neighbor_route(points)

    assert route[0] == 0
    assert sorted(route) == [0, 1, 2, 3]


def test_simulated_annealing_returns_history_and_best_state():
    points = {
        "A": (0.0, 0.0),
        "B": (0.0, 1.0),
        "C": (1.0, 1.0),
        "D": (1.0, 0.0),
        "E": (0.5, 0.2),
    }
    distance = lambda a, b: euclidean_distance(points[a], points[b])
    initial_route = ["A", "C", "B", "D", "E"]
    initial_cost = route_cost(initial_route, distance)

    result = simulated_annealing(
        initial_state=initial_route,
        objective=lambda route: route_cost(route, distance),
        neighbor=tsp_segment_neighbor,
        seed=4,
        init_temp=5.0,
        cooling_rate=0.95,
        max_iters=80,
        minimize=True,
    )

    assert set(result["best_state"]) == set(initial_route)
    assert len(result["history"]) == 80
    assert result["best_score"] <= initial_cost + 1e-9
    assert 0.0 <= result["accepted_worse_rate"] <= 1.0


def test_pareto_front_removes_dominated_records():
    records = [
        {"name": "fast", "duration": 8, "dose": 15},
        {"name": "balanced", "duration": 10, "dose": 10},
        {"name": "low-dose", "duration": 14, "dose": 7},
        {"name": "dominated", "duration": 12, "dose": 16},
    ]

    front = pareto_front(records, objectives=["duration", "dose"])
    names = {row["name"] for row in front}

    assert names == {"fast", "balanced", "low-dose"}
    assert "dominated" not in names


def test_euclidean_distance_is_plain_geometry():
    assert math.isclose(euclidean_distance((0, 0), (3, 4)), 5.0)

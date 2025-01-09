from .deterministic import (
    undirected_constraints,
    directed_constraints,
    add_flow_constraints,
    make_model
)
from .stochastic import add_previous_result
from .objects import (
    SSFP,
    Solution,
    Pipe
)
import gurobipy as gp
from gurobipy import GRB
import time
from typing import Dict, Tuple


def add_constraints(model: gp.Model, ssfp: SSFP) -> gp.Model:
    """
    Adds RO constraints to the model.
    :param model: Gurobi model.
    :param ssfp: SSFP-object.
    :return: Gurobi model with RO constraints.
    """
    # Add dummy decision variable
    d = model.addVar(vtype=GRB.CONTINUOUS, name="d")

    # Constraint 1: connect the dummy variable to the second stage costs.
    for scenario in ssfp.future:
        left_hand_side = 0
        for u, v in ssfp.all_edges:
            for p in ssfp.all_pipes:
                variable_name1 = f"{scenario.id}_x[{p.id},{u},{v}]"
                variable_name2 = f"{ssfp.present.id}_x[{p.id},{u},{v}]"
                variable1 = model.getVarByName(variable_name1)
                variable2 = model.getVarByName(variable_name2)
                cost_parameter = ssfp.get_second_stage_weight((u, v), p, scenario)
                left_hand_side += (variable1 - variable2) * cost_parameter
        model.addConstr(left_hand_side <= d, name="RO1")

    # Constraint 2: connect the first and second stage decision variables x.
    for scenario in ssfp.future:
        for u, v in ssfp.all_edges:
            for p in ssfp.all_pipes:
                variable_name1 = f"{scenario.id}_x[{p.id},{u},{v}]"
                variable_name2 = f"{ssfp.present.id}_x[{p.id},{u},{v}]"
                variable1 = model.getVarByName(variable_name1)
                variable2 = model.getVarByName(variable_name2)
                model.addConstr(variable1 >= variable2, name="RO2")

    model.update()

    return model


def objective(model: gp.Model, ssfp: SSFP) -> gp.Model:
    """
    Sets the objective of the robust models.
    :param model: Gurobi model.
    :param ssfp: SSFP-object.
    :return: Gurobi model with RO objective.
    """
    # Set objective
    term1 = 0
    for u, v in ssfp.all_edges:
        for p in ssfp.all_pipes:
            variable_name = f"{ssfp.present.id}_x[{p.id},{u},{v}]"
            variable = model.getVarByName(variable_name)
            cost_parameter = ssfp.get_first_stage_weight((u, v), p)
            term1 += variable * cost_parameter
    term2 = model.getVarByName("d")

    model.setObjective(term1 + term2, GRB.MINIMIZE)

    return model


def undirected_flow(
    ssfp: SSFP, time_limit: float, logfile: str
) -> Solution:
    """
    Solves the RO-U model.
    :param ssfp: SSFP-object.
    :param time_limit: time limit in seconds for the Gurobi model.
    :param logfile: path to logfile.
    :return: Solution-object.
    """

    # Create the model
    name = "Robust Undirected"
    model = make_model(name, time_limit, logfile)

    # Start tracking compilation time
    start_time = time.time()

    # Add constraints
    for scenario in [ssfp.present] + ssfp.future:
        model = undirected_constraints(model, scenario)

    # Add RO constraints
    model = add_constraints(model, ssfp)

    # Set objective
    model = objective(model, ssfp)

    # End tacking compilation time
    end_time = time.time()
    compilation_time = end_time - start_time

    # Optimize model
    model.optimize()

    # Generate solution object
    solution = Solution(model, ssfp, compilation_time, name)

    return solution


def directed_flow(
    ssfp: SSFP,
    time_limit: float,
    logfile: str,
    previous_result: Dict[Pipe, Dict[Tuple[int, int], int]] = None,
) -> Solution:
    """
    Solves the RO-D model.
    :param ssfp: SSFP-object.
    :param time_limit: time limit in seconds for the Gurobi model.
    :param logfile: path to logfile.
    :param previous_result: nested list with 1 if pipe p uses edge e and 0 otherwise.
    :return: Solution-object.
    """

    # Create the model
    name = "Robust Directed"
    model = make_model(name, time_limit, logfile)

    # Start tracking compilation time
    start_time = time.time()

    # Add constraints
    for scenario in [ssfp.present] + ssfp.future:
        model = directed_constraints(model, scenario)
        model = add_flow_constraints(model, scenario)

    # Add RO constraints
    model = add_constraints(model, ssfp)

    # Previous result
    model = add_previous_result(model, ssfp, previous_result)

    # Set objective
    model = objective(model, ssfp)

    # End tacking compilation time
    end_time = time.time()
    compilation_time = end_time - start_time

    # Optimize model
    model.optimize()

    # Generate solution object
    solution = Solution(model, ssfp, compilation_time, name)

    return solution

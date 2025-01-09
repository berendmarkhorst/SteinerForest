from .deterministic import (
    undirected_constraints,
    directed_constraints,
    add_flow_constraints,
    make_model,
    callback_scenario
)
import gurobipy as gp
from gurobipy import GRB
import time
from .objects import (
    SSFP,
    Pipe,
    Solution,
)
from typing import Tuple, Dict


def add_previous_result(
    model: gp.Model, ssfp: SSFP, previous_result: Dict[Pipe, Dict[Tuple[int, int], int]], variables: Dict[int, Tuple[gp.Var]],
) -> gp.Model:
    """
    We fix the solution to the previous result.
    :param model: Gurobi model.
    :param ssfp: SSFP-object.
    :param previous_result: nested list with 1 if pipe p uses edge e and 0 otherwise.
    :return: Gurobi model with previous result.
    """
    if previous_result is not None:
        x, *_ = variables[ssfp.present.id]
        for p in ssfp.all_pipes:
            for u, v in ssfp.all_edges:
                expr = x[p.id, u, v] == round(previous_result[p][(u, v)])
                model.addConstr(expr, name="previous result")

    return model


def constraints(model: gp.Model, ssfp: SSFP, variables: Dict[int, Tuple[gp.Var]]) -> gp.Model:
    """
    Adds SO constraints to the model.
    :param model: Gurobi model.
    :param ssfp: SSFP-object.
    :return: Gurobi model with SP constraints.
    """
    # Add constraints
    for scenario in ssfp.future:
        x_present, *_ = variables[ssfp.present.id]
        x, *_ = variables[scenario.id]
        for u, v in ssfp.all_edges:
            for p in ssfp.all_pipes:
                variable1 = x[p.id, u, v]
                variable2 = x_present[p.id, u, v]
                model.addConstr(variable1 >= variable2, name="SO")

    return model


def objective(model: gp.Model, ssfp: SSFP, variables: Dict[int, Tuple[gp.Var]]) -> gp.Model:
    """
    Sets the objective of the stochastic models.
    :param model: Gurobi model.
    :param ssfp: SSFP-object.
    :return: Gurobi model with SO objective.
    """
    x_present, *_ = variables[ssfp.present.id]
    first_stage_costs = 0
    for p in ssfp.all_pipes:
        for u, v in ssfp.all_edges:
            variable = x_present[p.id, u, v]
            cost_parameter = ssfp.graph.edges()[(u, v)][
                f"weight first stage pipe {p.id}"
            ]
            first_stage_costs += variable * cost_parameter

    second_stage_costs = 0
    for scenario in ssfp.future:
        x, *_ = variables[scenario.id]
        for u, v in ssfp.all_edges:
            for p in ssfp.all_pipes:
                variable1 = x[p.id, u, v]
                variable2 = x_present[p.id, u, v]
                cost_parameter = ssfp.graph.edges()[(u, v)][
                    f"weight second stage scenario {scenario.id} pipe {p.id}"
                ]
                second_stage_costs += (
                    (variable1 - variable2) * cost_parameter * scenario.probability
                )

    model.setObjective(first_stage_costs + second_stage_costs, GRB.MINIMIZE)

    return model


def undirected_flow(ssfp: SSFP, time_limit: float, logfile: str):
    """
    Solves the SP-U model.
    :param ssfp: SSFP-object.
    :param time_limit: time limit in seconds for the Gurobi model.
    :param logfile: path to logfile.
    :return: Solution-object.
    """

    # Create the model
    name = "Stochastic Undirected"
    model = make_model(name, time_limit, logfile)

    # Start tracking compilation time
    start_time = time.time()

    # Add constraints
    for scenario in [ssfp.present] + ssfp.future:
        model = undirected_constraints(model, scenario)

    # Add SO constraints
    model = constraints(model, ssfp)

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
    node_file: str = None,
    previous_result: Dict[Pipe, Dict[Tuple[int, int], int]] = None,
    warm_start_values: Dict = None,
) -> Solution:
    """
    Solves the SO-D model.
    :param ssfp: SSFP-object.
    :param time_limit: time limit in seconds for the Gurobi model.
    :param logfile: path to logfile.
    :param node_file: path to node file.
    :param previous_result: nested list with 1 if pipe p uses edge e and 0 otherwise.
    :return: Solution-object.
    """
    # Create the model
    name = "Stochastic Directed"
    model = make_model(name, time_limit, logfile, node_file=node_file)

    # Start tracking compilation time
    start_time = time.time()

    # Add constraints
    variables = {}
    for scenario in [ssfp.present] + ssfp.future:
        model, *common_variables = directed_constraints(model, scenario)
        model, flow_variables = add_flow_constraints(model, scenario)
        variables[scenario.id] = common_variables + [flow_variables]

    # Add SO constraints
    model = constraints(model, ssfp, variables)

    # Previous result
    model = add_previous_result(model, ssfp, previous_result, variables)

    # Warm start
    model = warm_start(model, warm_start_values)

    # Set objective
    model = objective(model, ssfp, variables)

    # End tacking compilation time
    end_time = time.time()
    compilation_time = end_time - start_time

    # Optimize model
    model.optimize()

    # Generate solution object
    solution = Solution(model, ssfp, compilation_time, name)

    return solution


def root_node_cut(model, ssfp):
    # Make sure the model does not get warm started by a previous solution.
    model.reset()

    # Make an empty _cuts attribute
    model._cuts = {}
    for scenario in [ssfp.present] + ssfp.future:
        model._cuts[scenario.id] = []

    # Set all variables to continuous
    for var in model.getVars():
        var.vtype = "C"
    model.update()

    def callback(model, where):
        if where == GRB.Callback.MIPSOL or (where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
            for scenario in [ssfp.present] + ssfp.future:
                callback_scenario(model, where, scenario)

    # Optimize model
    model.Params.lazyConstraints = 1
    model.optimize(callback)

    # Generate solution object
    solution = Solution(model, ssfp, 0, "Root Node")

    return solution


def directed_cut(
    ssfp: SSFP,
    time_limit: float,
    logfile: str,
    node_file: str = None,
    previous_result: Dict[Pipe, Dict[Tuple[int, int], int]] = None,
    warm_start_values: Dict = None,
    rootnode = False,
) -> Solution:
    """
    Solves the stochastic directed cut model.
    :param scenario: Scenario-object.
    :param time_limit: time limit in seconds for the Gurobi model.
    :param logfile: path to logfile.
    :param node_file: path to node file.
    :param previous_result: nested list with 1 if pipe p uses edge e and 0 otherwise.
    :return: Solution-object.
    """
    # Create the model
    name = "Stochastic Directed Cut"
    model = make_model(name, time_limit, logfile, node_file=node_file)

    # Start tracking compilation time
    start_time = time.time()

    # Add constraints
    variables = {}
    for scenario in [ssfp.present] + ssfp.future:
        model, *variables[scenario.id] = directed_constraints(model, scenario)
        scenario.add_digraph()

    # Add SO constraints
    model = constraints(model, ssfp, variables)

    # Previous result
    model = add_previous_result(model, ssfp, previous_result, variables)

    # Warm start
    model = warm_start(model, warm_start_values)

    # Save cuts for later
    model._cuts = []
    model._cuts_LHS = []
    model._cuts_RHS = []

    # Set objective
    model = objective(model, ssfp, variables)

    # End tacking compilation time
    end_time = time.time()
    compilation_time = end_time - start_time

    # Callback functions
    def callback(model, where):
        # Stop after the root node.
        if rootnode and where == GRB.Callback.MIP and model.cbGet(GRB.Callback.MIP_NODCNT) != 0:
            model.terminate()
            return

        if where == GRB.Callback.MIPSOL or (where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
            for scenario in [ssfp.present] + ssfp.future:
                callback_scenario(model, where, scenario, variables)

    # Optimize model
    model.Params.lazyConstraints = 1
    model.optimize(callback)

    # Generate solution object
    solution = Solution(model, ssfp, compilation_time, name)
    solution.variables = variables

    return solution
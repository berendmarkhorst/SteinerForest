from .model_helpers import (
    deduct_sets,
    demand_and_supply_undirected,
    demand_and_supply_directed,
    get_terminal_groups_until_k,
    terminal_groups_without_root,
    find_index_terminal
)
from .objects import Scenario, Solution
import networkx as nx
import time
import gurobipy as gp
from gurobipy import GRB
from typing import Union, Dict, Tuple


def make_model(name: str, time_limit: float, logfile: str, node_file: str="") -> gp.Model:
    """
    Creates a Gurobi model with the given name, time limit and logfile.
    :param name: name of the Gurobi model.
    :param time_limit: time limit in seconds for the Gurobi model.
    :param logfile: path to logfile.
    :param node_file: path to node file.
    :return: Gurobi model.
    """
    # Create model
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()

    # Make the model and set the time limit
    model = gp.Model(name, env=env)
    model.setParam("TimeLimit", time_limit)

    # Clear the logfile and start logging
    if logfile:
        with open(logfile, "w") as _:
            pass
        model.setParam("LogFile", logfile)

    # Set the model to single-threaded
    model.setParam('Threads', 1)

    # If node file is an empty string, we do not use node files. Otherwise, we do so after 0.5 GB memory is used.
    if node_file != "":
        model.setParam("NodeFileStart", 0.5)
        model.setParam("NodeFileDir", node_file)

    return model


def set_objective(model: gp.Model, scenario: Scenario) -> gp.Model:
    """
    Sets the objective of the deterministic models.
    :param model: Gurobi model.
    :param scenario: Scenario-object.
    :return: Gurobi model with objective.
    """
    expression = 0
    for p in scenario.pipes:
        for u, v in list(scenario.parent.graph.edges()):
            variable_name = f"{scenario.id}_x[{p.id},{u},{v}]"
            variable = model.getVarByName(variable_name)
            cost_parameter = scenario.parent.get_first_stage_weight((u, v), p)
            expression += variable * cost_parameter

    model.setObjective(expression, GRB.MINIMIZE)

    model.update()

    return model


def undirected_constraints(model: gp.Model, scenario: Scenario) -> gp.Model:
    """
    Adds DO-U constraints to the model.
    :param model: Gurobi model.
    :param scenario: Scenario-object.
    :return: Gurobi model with DO-U constraints.
    """
    # Deduct sets
    admissible_arcs, terminals, roots, vertices, _, all_pipes, admissible_pipes = (
        deduct_sets(scenario)
    )
    terminals_without_roots = set(terminals) - set(roots)

    # Decision variables
    f = model.addVars(
        terminals_without_roots,
        admissible_pipes,
        admissible_arcs,
        vtype=GRB.BINARY,
        name=f"{scenario.id}_f",
    )
    x = model.addVars(
        all_pipes, scenario.parent.all_edges, vtype=GRB.BINARY, name=f"{scenario.id}_x"
    )

    # Constraint 1: flow conservation
    for v in vertices:
        for t in terminals_without_roots:
            demand_and_supply = demand_and_supply_undirected(
                v, t, roots[find_index_terminal(scenario.terminal_groups, t)]
            )

            left_hand_side = 0
            right_hand_side = 0

            for p in scenario.pipes:
                left_hand_side += sum(
                    f[t, p.id, a[0], a[1]] for a in admissible_arcs if a[0] == v
                )
                right_hand_side += sum(
                    f[t, p.id, a[0], a[1]] for a in admissible_arcs if a[1] == v
                )
            expr = left_hand_side - right_hand_side == demand_and_supply
            model.addConstr(expr, name="1")

    # Constraint 2: connection between f and x
    for e in scenario.edges:
        for p in scenario.pipes:
            for t in terminals_without_roots:
                left_hand_side = f[t, p.id, e[0], e[1]] + f[t, p.id, e[1], e[0]]
                right_hand_side = x[p.id, e[0], e[1]]
                expr = left_hand_side <= right_hand_side
                model.addConstr(expr, name="2")

    model.update()

    return model


def undirected_flow(
    scenario: Scenario, time_limit: float, logfile: str
) -> Solution:
    """
    Solves the deterministic undirected model.
    :param scenario: Scenario-object.
    :param time_limit: time limit in seconds for the Gurobi model.
    :param logfile: path to logfile.
    :return: Solution-object.
    """
    # Create the model
    name = "Deterministic Undirected with Flows"
    model = make_model(name, time_limit, logfile)

    # Start tracking compilation time
    start_time = time.time()

    # Add constraints
    model = undirected_constraints(model, scenario)

    # Set objective
    model = set_objective(model, scenario)

    # End tacking compilation time
    end_time = time.time()
    compilation_time = end_time - start_time

    # Optimize model
    model.optimize()

    # Generate solution object
    solution = Solution(model, scenario.parent, compilation_time, name)

    return solution


def directed_constraints(model: gp.Model, scenario: Scenario) -> Union[gp.Model, gp.Var]:
    """
    Adds DO-D constraints to the model.
    :param model: Gurobi model.
    :param scenario: Scenario-object.
    :return: Gurobi model with DO-D constraints and decision variables.
    """
    # Sets
    arcs, terminals, roots, vertices, k_list, all_pipes, pipes = deduct_sets(scenario)
    k_indices = [(k, l) for k in k_list for l in k_list if l >= k]
    steiner_points = set(vertices) - set(terminals)

    # Decision variables
    x = model.addVars(
        all_pipes, scenario.parent.all_edges, vtype=GRB.BINARY, name=f"{scenario.id}_x"
    )
    y1 = model.addVars(pipes, arcs, vtype=GRB.BINARY, name=f"{scenario.id}_y1")
    y2 = model.addVars(k_list, pipes, arcs, vtype=GRB.BINARY, name=f"{scenario.id}_y2")
    z = model.addVars(k_indices, vtype=GRB.BINARY, name=f"{scenario.id}_z")

    # Constraint 1: connection between y2 and y1
    # Corresponds with (6c) from Schmidt.
    for u, v in arcs:
        for p in scenario.pipes:
            left_hand_side = gp.quicksum(y2[k, p.id, u, v] for k in k_list)
            right_hand_side = y1[p.id, u, v]
            expr = left_hand_side <= right_hand_side
            model.addConstr(expr, name="1")

    # Constraint 2: indegree of each vertex cannot exceed 1
    # Corresponds with (6f) from Schmidt.
    for v in vertices:
        left_hand_side = 0
        for p in scenario.pipes:
            for u, w in arcs:
                if v == w:
                    left_hand_side += y1[p.id, u, w]
        expr = left_hand_side <= 1
        model.addConstr(expr, name="2")

    # Constraint 3: connection between y1 and x
    # Corresponds with (6e) from Schmidt.
    for u, v in scenario.edges:
        for p in scenario.pipes:
            expr = y1[p.id, u, v] + y1[p.id, v, u] <= x[p.id, u, v]
            model.addConstr(expr, name="3")

    # Constraint 4: we enforce that every terminal group is rooted at exactly one root
    # Corresponds with (6b) from Schmidt.
    for k in k_list:
        left_hand_side = gp.quicksum(z[l, k] for l in range(k + 1))
        right_hand_side = 1
        expr = left_hand_side == right_hand_side
        model.addConstr(expr, name="4")

    # Constraint 5: we enforce exactly one root in each arborescence.
    # Corresponds with (6d) from Schmidt.
    for k in k_list[1:-1]:
        for l in k_list:
            if l > k:
                left_hand_side = z[k, k]
                right_hand_side = z[k, l]
                expr = left_hand_side >= right_hand_side
                model.addConstr(expr, name="5")

    # Constraint 6: the terminals in T^{1···k−1} cannot be attached to root r k and thus,
    # no arc of the corresponding arborescence should enter such a terminal
    # Corresponds with (6g) from Schmidt.
    for k in k_list[1:]:
        for t in get_terminal_groups_until_k(scenario.terminal_groups, k):
            left_hand_side = 0
            for p in scenario.pipes:
                for u, v in arcs:
                    if v == t:
                        left_hand_side += y2[k, p.id, u, v]
            expr = left_hand_side == 0
            model.addConstr(expr, name="6")

    # Constraint 7: we enforce that the indegree of a vertex is at most the outdegree for the Steiner points in the
    # overall solution.
    # Corresponds with (9a) from Schmidt.
    expression = []
    for v in steiner_points:
        left_hand_side = gp.quicksum(
            y1[p.id, a[0], a[1]] for p in scenario.pipes for a in arcs if a[1] == v
        )
        right_hand_side = gp.quicksum(
            y1[p.id, a[0], a[1]] for p in scenario.pipes for a in arcs if a[0] == v
        )
        expr = left_hand_side <= right_hand_side
        expression.append(expr)
        model.addConstr(expr, name="7")

    # Constraint 8: we enforce that the indegree of a vertex is at most the outdegree for the Steiner points in
    # each terminal group.
    # Corresponds with (9b) from Schmidt.
    for k in k_list:
        remaining_vertices = set(vertices) - set(
            terminal_groups_without_root(scenario.terminal_groups, roots, k)
        )
        for v in remaining_vertices:
            left_hand_side = gp.quicksum(
                y2[k, p.id, a[0], v] for p in scenario.pipes for a in arcs if a[1] == v
            )
            right_hand_side = gp.quicksum(
                y2[k, p.id, v, a[1]] for p in scenario.pipes for a in arcs if a[0] == v
            )
            expr = left_hand_side <= right_hand_side
            expression.append(expr)
            model.addConstr(expr, name="8")

    # Constraint 9: connect y2 and z.
    # Corresponds with (9c) from Schmidt.
    for p in scenario.pipes:
        for k in k_list:
            for l in k_list:
                if l > k:
                    left_hand_side = gp.quicksum(
                        y2[k, p.id, a[0], a[1]] for a in arcs if a[1] == roots[l]
                    )
                    right_hand_side = z[k, l]
                    expr = left_hand_side <= right_hand_side
                    model.addConstr(expr, name="9")

    model.update()

    return model, x, y1, y2, z


def add_flow_constraints(model: gp.Model, scenario: Scenario) -> gp.Model:
    """
    We add the flow constraints to the gurobipy model.
    :param model: Gurobi model.
    :param scenario: Scenario-object.
    :return: Gurobi model and variable(s).
    """
    # Sets
    arcs, terminals, roots, vertices, k_list, all_pipes, pipes = deduct_sets(scenario)
    f_indices = [
        (k, t, p.id, a[0], a[1])
        for k in k_list
        for t in terminal_groups_without_root(scenario.terminal_groups, roots, k)
        for p in scenario.pipes
        for a in arcs
    ]

    # Decision variables
    f = model.addVars(f_indices, vtype=GRB.BINARY, name=f"{scenario.id}_f")

    # Constraint 1: flow conservation
    # Corresponds with (7b) from Schmidt.
    for v in vertices:
        for k in k_list:
            for t in terminal_groups_without_root(scenario.terminal_groups, roots, k):
                left_hand_side = 0
                for p in scenario.pipes:
                    first_term = sum(
                        f[k, t, p.id, a[0], a[1]] for a in arcs if a[0] == v
                    )
                    second_term = sum(
                        f[k, t, p.id, a[0], a[1]] for a in arcs if a[1] == v
                    )
                    left_hand_side += first_term - second_term
                demand_and_supply = demand_and_supply_directed(
                    model, scenario, t, v, roots, k
                )
                expr = left_hand_side == demand_and_supply
                model.addConstr(expr, name="flow 1")

    # Constraint 2: connection between f and y2
    # Corresponds with (7a) from Schmidt.
    for k in k_list:
        for t in terminal_groups_without_root(scenario.terminal_groups, roots, k):
            for u, v in arcs:
                for p in scenario.pipes:
                    left_hand_side = f[k, t, p.id, u, v]
                    variable_name = f"{scenario.id}_y2[{k},{p.id},{u},{v}]"
                    right_hand_side = model.getVarByName(variable_name)
                    expr = left_hand_side <= right_hand_side
                    model.addConstr(expr, name="flow 2")

    # Constraint 3: we prevent a flow from leaving a terminal.
    # Corresponds with (7c) from Schmidt.
    for k in k_list:
        for t in terminal_groups_without_root(scenario.terminal_groups, roots, k):
            left_hand_side = 0
            for p in scenario.pipes:
                for u, v in arcs:
                    if u == t:
                        left_hand_side += f[k, t, p.id, u, v]
            expr = left_hand_side == 0
            model.addConstr(expr, name="flow 3")

    model.update()

    return model, f


def directed_flow(
    scenario: Scenario, time_limit: float, logfile: str
) -> Solution:
    """
    Solves the deterministic directed model.
    :param scenario: Scenario-object.
    :param time_limit: time limit in seconds for the Gurobi model.
    :param logfile: path to logfile.
    :return: Solution-object.
    """
    # Create the model
    name = "Deterministic Directed"
    model = make_model(name, time_limit, logfile)

    # Start tracking compilation time
    start_time = time.time()

    # Add constraints
    model = directed_constraints(model, scenario)
    model = add_flow_constraints(model, scenario)

    # Set objective
    model = set_objective(model, scenario)

    # End tacking compilation time
    end_time = time.time()
    compilation_time = end_time - start_time

    # Optimize model
    model.optimize()

    # Generate solution object
    solution = Solution(model, scenario.parent, compilation_time, name)

    return solution


def directed_cut(
    scenario: Scenario, time_limit: float, logfile: str
) -> Solution:
    """
    Solves the deterministic directed model.
    :param scenario: Scenario-object.
    :param time_limit: time limit in seconds for the Gurobi model.
    :param logfile: path to logfile.
    :return: Solution-object.
    """
    # Create the model
    name = "Deterministic Directed Cut"
    model = make_model(name, time_limit, logfile)

    # Start tracking compilation time
    start_time = time.time()

    # Add constraints
    model = directed_constraints(model, scenario)

    # Add digraph to the scenario. Will be useful in the user-callbacks.
    scenario.add_digraph()

    # Set objective
    model = set_objective(model, scenario)

    # End tacking compilation time
    end_time = time.time()
    compilation_time = end_time - start_time

    # Callback functions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL or (where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
            callback_scenario(model, where, scenario)

    # Optimize model
    model.Params.lazyConstraints = 1
    model.optimize(callback)

    # Generate solution object
    solution = Solution(model, scenario.parent, compilation_time, name)

    return solution


def callback_scenario(model: gp.Model, where, scenario: Scenario, variables: Dict[int, Tuple[gp.Var]]):
    """
    Callback function for the directed cut model.
    :param model: Gurobi model.
    :param where: contains the position in the solving process.
    :param scenario: Scenario-object.
    :param variables: Gurobi variables.
    """

    # Skip if there are no terminal groups.
    if len(scenario.terminal_groups[0]) == 0:
        return

    # Constants
    eps = 1e-6

    # Variables
    x, y1, y2, z = variables[scenario.id]

    for index_l, group_l in enumerate(scenario.terminal_groups):
        root_l = group_l[0]

        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(y2)
        elif where == GRB.Callback.MIPNODE:
            vals = model.cbGetNodeRel(y2)

        for index_k, group_k in enumerate(scenario.terminal_groups[: index_l + 1]):
            root_k = group_k[0]

            for (u, v) in scenario.arcs:
                # Creep flows, see Schmidt 4.1
                scenario.digraph.edges()[(u, v)]["capacity"] = sum(vals[index_k, p.id, u, v] for p in scenario.pipes) + eps

            if where == GRB.Callback.MIPSOL:
                z_value = model.cbGetSolution(z[index_k, index_l])
            elif where == GRB.Callback.MIPNODE:
                z_value = model.cbGetNodeRel(z[index_k, index_l])

            for t in group_l:
                if root_k != t:
                    # Get minimum cut (or maximum flow) value
                    cut_value, partition = nx.minimum_cut(
                        scenario.digraph, root_k, t, capacity="capacity"
                    )

                    if cut_value < z_value:
                        cut_arcs = [
                            (u, v)
                            for (u, v) in scenario.arcs
                            if u in partition[0] and v in partition[1]
                        ]

                        for (u, v) in cut_arcs:
                            scenario.digraph.edges()[(u, v)]["capacity"] = 1

                        left_hand_side_list = []
                        for p in scenario.pipes:
                            for (u, v) in cut_arcs:
                                left_hand_side_list.append(y2[index_k, p.id, u, v])
                        right_hand_side = z[index_k, index_l]

                        cut = gp.quicksum(left_hand_side_list) >= right_hand_side

                        model.cbLazy(cut)

                        model._cuts.append(cut)
                        model._cuts_LHS.append(left_hand_side_list)
                        model._cuts_RHS.append(right_hand_side)

                    # Back cuts
                    residual_graph = nx.algorithms.flow.build_residual_network(scenario.digraph, cut_value)

                    cut_value, partition = nx.minimum_cut(residual_graph, root_k, t, capacity="capacity")

                    cut_arcs = [(u, v) for (u, v) in scenario.arcs
                                if u in partition[0] and v in partition[1]]

                    left_hand_side_list = []
                    for p in scenario.pipes:
                        for (u, v) in cut_arcs:
                            left_hand_side_list.append(y2[index_k, p.id, u, v])
                    right_hand_side = z[index_k, index_l]

                    cut = gp.quicksum(left_hand_side_list) >= right_hand_side

                    model.cbLazy(cut)

                    model._cuts.append(cut)
                    model._cuts_LHS.append(left_hand_side_list)
                    model._cuts_RHS.append(right_hand_side)
import numpy as np
from .objects import Scenario, SSFP, Pipe, Solution
import src.stochastic as so
import src.deterministic as do
from gurobipy import GRB
import gurobipy as gp
import copy
import time


def distance(ssfp, scenario1, scenario2, alpha1=1, alpha2=0):
    """
    Computes the distance between scenario1 and scenario2.
    :param ssfp: SSFP-object.
    :param scenario1: Scenario-object.
    :param scenario2: Scenario-object.
    :param alpha1: float between 0 and 1, weight for distance metric L1.
    :param alpha2: float between 0 and 1, weight for distance metric L2.
    :return: distance between scenario1 and scenario2.
    """
    alpha3 = 1 - alpha1 - alpha2

    terminal_difference = 0
    for k1 in scenario1.terminal_groups:
        differences = []
        for k2 in scenario2.terminal_groups:
            difference1 = set(k1).difference(set(k2))
            difference2 = set(k2).difference(set(k1))
            total_difference = difference1.union(difference2)
            differences.append(len(total_difference))
        terminal_difference += min(differences)

    weights1 = np.array([ssfp.graph.edges()[edge][f"weight second stage scenario {scenario1.id} pipe 1"]
                         for edge in ssfp.graph.edges()])
    weights2 = np.array([ssfp.graph.edges()[edge][f"weight second stage scenario {scenario2.id} pipe 1"]
                         for edge in ssfp.graph.edges()])

    difference1 = set(scenario1.pipes).difference(set(scenario2.pipes))
    difference2 = set(scenario2.pipes).difference(set(scenario1.pipes))
    total_difference = difference1.union(difference2)
    media_difference = len(total_difference)

    return alpha1 * np.sum((weights1 - weights2)**2)**0.5 + alpha2 * terminal_difference + alpha3 * media_difference


def fast_forward_selection(ssfp, a_final, alpha1=1, alpha2=0):
    """
    Applies fast forward selection from (Heitsch and Römisch, 2003)
    :param ssfp: SSFP-object.
    :param a_final: number of scenarios to be included in the reduced subset.
    :param alpha1: weight for distance metric L1.
    :param alpha2: weight for distance metric L2.
    :return: a list of scenarios.
    """
    a = list()

    delta = {frozenset((k.id, l.id)): distance(ssfp, k, l, alpha1=alpha1, alpha2=alpha2) for i, k in enumerate(ssfp.future) for l in ssfp.future[:i+1]} # Dit kan sneller vanwege symmetrie.
    z = np.array([sum(k.probability * delta[frozenset((k.id, l.id))] for k in ssfp.future if k.id != l.id) for l in ssfp.future])
    u = ssfp.future[np.argmin(z)]
    j_set = [scenario for scenario in ssfp.future if scenario.id != u.id]
    a.append(u)

    for t in range(1, a_final):
        delta = {frozenset((k.id, l.id)): np.minimum(delta[frozenset((k.id, l.id))], delta[frozenset((k.id, u.id))]) for i, k in enumerate(j_set) for l in j_set[:i + 1]}
        z = np.array([sum(k.probability * delta[frozenset((k.id, l.id))] for k in j_set if k.id != l.id) for l in j_set])
        u = j_set[np.argmin(z)]
        a.append(u)
        j_set.remove(u)

    return a


def divide_over_a(ssfp, a, alpha1=1, alpha2=0):
    """
    Divides the scenarios over the scenarios in a.
    :param ssfp: SSFP-object.
    :param a: list of selected scenarios.
    :param alpha1: weight for distance metric L1.
    :param alpha2: weight for distance metric L2.
    :return: dictionary with selected scenarios as keys and a list of scenario ids as values.
    """
    groups = {j.id: [] for j in a}

    for i in ssfp.future:
        distances = [distance(ssfp, i, j, alpha1=alpha1, alpha2=alpha2) for j in a]
        group_index = a[np.argmin(distances)].id
        groups[group_index].append(i.id)

    return groups


def make_aggregated_problem(ssfp, groups):
    """
    Makes an aggregated problem from the groups.
    :param ssfp: SSFP-object.
    :param groups: dictionary with selected scenarios as keys and a list of scenario ids as values.
    :return: SSFP-object.
    """
    future = []

    for index, group in groups.items():
        probability = sum(ssfp.future[id - 1].probability for id in group)

        scenario = copy.deepcopy(ssfp.future[index - 1])
        scenario.probability = probability

        future += [scenario]

    ssfp_agg = SSFP(ssfp.graph, ssfp.all_pipes, ssfp.present, future)

    return ssfp_agg


def solve_with_scenario_selection(ssfp, a_final, output, node_file=None):
    """
    Solves the SSFP with TULIP.
    :param ssfp: SSFP-object.
    :param a_final: number of scenarios to be included in the reduced subset.
    :param output: log_file for Gurobi output.
    :param node_file: directory for node-file.
    :return: Solution-object and time it takes for the first step of TULIP.
    """
    start_time = time.time()
    a = fast_forward_selection(ssfp, a_final)
    groups = divide_over_a(ssfp, a)
    ssfp_agg = make_aggregated_problem(ssfp, groups)
    new_scenarios_included = [scenario.id for scenario in ssfp.future if scenario.id not in groups.keys()]
    end_time = time.time()
    scenario_selection_time = end_time - start_time
    maximum_runtime = 7200 - scenario_selection_time

    solution_aggregated = so.directed_cut(ssfp_agg, maximum_runtime, output, node_file=node_file)
    model = solution_aggregated.model
    first_iteration_time = time.time() - start_time

    # If the first iteration takes over two hours to solve, we don't have time for a second iteration.
    if first_iteration_time > 7200:
        return None

    solution = extend_model(ssfp, model, new_scenarios_included, model.Runtime + scenario_selection_time, solution_aggregated.variables, solution_aggregated.compilation_time)
    solution.first_iteration_time = first_iteration_time

    return solution, scenario_selection_time

def extend_model(ssfp, model, new_scenarios_included, wall_time_passed, variables, previous_compilation_time, rootnode=False, cuts=True, values=True):
    """
    Second step from the TULIP approach.
    :param ssfp: SSFP-object.
    :param model: Gurobi model.
    :param new_scenarios_included: extra scenarios to be included.
    :param wall_time_passed: time used in the first step from the TULIP approach.
    :param variables: Gurobi variables.
    :param previous_compilation_time: compulation time used in the first step from the TULIP approach.
    :param rootnode: boolean indicating if we only solve the root node in the first step of the TULIP approach.
    :param cuts: boolean indicating if we use the cuts made in the first step of the TULIP approach.
    :param values: boolean indicating if we use the solution from the first step of the TULIP approach.
    :return: Solution-object.
    """
    name = "Scenario Selection"

    start = time.time()
    # If there were cuts from the previous model, add them as constraints here.
    tight_cut_counter = 0
    model._cut_counter = len(model._cuts)
    if cuts:
        for idx, cut_LHS in enumerate(model._cuts_LHS):
            cut_RHS = model._cuts_RHS[idx]
            if abs(sum(var.X for var in cut_LHS) - cut_RHS.X) < 1e-6:
                model.addConstr(model._cuts[idx])
                tight_cut_counter += 1
        model.update()

    model._tight_cut_counter = tight_cut_counter
    model._cuts = []
    model._cuts_LHS = []
    model._cuts_RHS = []

    # We have already used some of our time for the warm start.
    model.setParam('TimeLimit', max(7200 - wall_time_passed, 0.01))

    # If you don't want to use the values from the previous model as warm start, reset the model.
    if not values:
        model.reset()

    for scenario in ssfp.future:
        if scenario.id in new_scenarios_included:
            model, *variables[scenario.id] = do.directed_constraints(model, scenario)

    for scenario in [ssfp.present] + ssfp.future:
        scenario.add_digraph()

    # Connect first and second stage variables with each other.
    model = so.constraints(model, ssfp, variables)

    # Set objective original SSFP
    model = so.objective(model, ssfp, variables)

    # Callback functions
    def callback(model, where):
        # Stop after the root node.
        if rootnode and where == GRB.Callback.MIP and model.cbGet(GRB.Callback.MIP_NODCNT) != 0:
            model.terminate()

        if where == GRB.Callback.MIPSOL or (where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
            for scenario in [ssfp.present] + ssfp.future:
                do.callback_scenario(model, where, scenario, variables)

    # Compute compilation time
    end_time = time.time()
    compilation_time = end_time - start + previous_compilation_time

    # Optimize model
    model.Params.lazyConstraints = 1
    model.optimize(callback)

    # Generate solution object
    solution = Solution(model, ssfp, compilation_time, name)

    return solution


def multiple_iterations(ssfp, a_list, output):
    """
    We apply TULIP multiple times. More specifically, for every number of scenarios in a_list.
    :param ssfp: SSFP-object.
    :param a_list: list of integers.
    :param output: log-file.
    :return: Solution-object.
    """
    for idx in range(len(a_list)):
        a_final = a_list[idx]
        a = fast_forward_selection(ssfp, a_final)
        groups = divide_over_a(ssfp, a)
        ssfp_agg = make_aggregated_problem(ssfp, groups)

        # Misschien is dit niet eens nodig!
        if idx == len(a_list) - 1:
            tolerance = 0.01
        else:
            tolerance = 0.05

        if idx == 0:
            solution_aggregated = so.directed_cut(ssfp_agg, 7200, output)
            model = solution_aggregated.model
            wall_time_passed = model.Runtime
        else:
            included_scenario_ids = [scenario.id for scenario in previous_ssfp_agg.future]
            solution = extend_model(ssfp_agg, model, included_scenario_ids, wall_time_passed, tolerance)
            wall_time_passed += solution.model.Runtime
        previous_ssfp_agg = ssfp_agg

    solution.steps = a_list

    return solution

def get_distance_matrix(ssfp):
    """
    Computes the distance matrix for all scenarios within ssfp.
    :param ssfp: SSFP-object.
    :return: distance matrix.
    """
    delta = {}
    for i, scenario1 in enumerate(ssfp.future):
        for scenario2 in ssfp.future:
            delta[(scenario1.id, scenario2.id)] = distance(ssfp, scenario1, scenario2, alpha1=1, alpha2=0)
    return delta


def find_closest_distance(a, ssfp, distance_matrix):
    """
    Computes the closest distance between a and all scenarios in ssfp.
    :param a: list of scenario-objects.
    :param ssfp: SSFP-object.
    :param distance_matrix: distance matrix.
    :return: dictionary with closest distances per scenario.
    """
    distance_dict = {}
    for scenario in ssfp.future:
        closest_distance = min([distance_matrix[(scenario.id, s.id)] for s in a])
        distance_dict[scenario] = closest_distance
    return distance_dict
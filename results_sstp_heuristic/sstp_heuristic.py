import benchmark_dimacs as bd
import networkx as nx
import itertools

from src.deterministic import directed_flow
from src.objects import SSFP, Scenario, Pipe
import src.deterministic as do
import src.stochastic as sp
import numpy as np
import copy
import time
import os
from typing import List, Dict, Tuple
import argparse
from networkx.algorithms.approximation import steinertree
import random
import math
from sklearn.cluster import DBSCAN
from collections import Counter


def ssfp_to_sstp(ssfp: SSFP):
    """
    This function converts a SSFP instance to a SSTP instance. This means that we only have one pipe type and that we only
    have one terminal group for each scenario.
    """

    # From a forest to a tree
    ssfp.present.terminal_groups = [[]]
    for scenario in ssfp.future:
        scenario.terminal_groups = [scenario.terminal_groups[-1]]

    # One pipe type
    only_pipe = ssfp.all_pipes[0]
    ssfp.all_pipes = [only_pipe]

    for scenario in [ssfp.present] + ssfp.future:
        scenario.pipes = ssfp.all_pipes

    return ssfp


def update_score(score_dict, scenario, edges):
    """
    This function updates two variables from heuristic1 for a given scenario based on the first k paths.
    :return:
    """
    # Update the score_dict per edge
    for edge in edges:
        # Prevent issues with swapped vertices!
        if edge in score_dict[scenario.id].keys():
            new_edge = edge
        else:
            new_edge = (edge[1], edge[0])
        score_dict[scenario.id][frozenset(new_edge)] = 1


    return score_dict

def take_subset(ssfp: SSFP, subset_ratio: float):
    """
    Take a subset of the scenario tree or return the scenario tree itself.
    :param ssfp: SSFP object.
    :param subset_ratio: ratio of the scenario tree that is captured in the sample.
    :return: (subset of) the scenario tree.
    """
    if subset_ratio < 1:
        subset_size = math.ceil(subset_ratio * len(ssfp.future))
        subset_scenarios = random.sample(ssfp.future, subset_size)
    else:
        subset_scenarios = ssfp.future

    return subset_scenarios

def compute_exact_steiner_tree(ssfp, dummy_ssfp, scenario, weight: str, node_files: str):
    """
    We compute the exact steiner tree solution and objective for the ssfp graph and the given scenario.
    :param ssfp: SSFP object.
    :param dummy_ssfp: dummy SSFP object.
    :param scenario: scenario object.
    :param weight: attribute name of weight that should be used.
    :return: objective and the exact steiner tree.
    """
    for edge in ssfp.graph.edges():
        dummy_ssfp.graph.edges[edge]["weight first stage pipe 1"] = ssfp.graph.edges[edge][weight]

    dummy_ssfp.present.terminal_groups = scenario.terminal_groups

    # # Get the heuristic solution...
    # heuristic_objective, heuristic_solution = compute_steiner_tree(ssfp.graph, scenario.terminal_groups[0], weight)

    # ... which can serve as a warmstart for the ILP.
    solution = do.directed_cut(dummy_ssfp.present, 3600, "log.txt", node_file=node_files)
    objective = solution.objective
    route = [k for k, v in solution.route[dummy_ssfp.all_pipes[0]].items() if v == 1]

    return objective, route

def compute_steiner_tree(graph: nx.Graph, terminals: List, weight: str):
    """
    We compute the (approximate) steiner tree for the given graph, weight, and terminal-set.
    :param ssfp: SSFP object.
    :param terminals: list of terminals.
    :param weight: string representing the weight of the edges
    :return: objective and the selected edges (as a list).
    """
    # Compute the approximate Steiner tree to add the second stage costs for this specific terminal-set.
    T = steinertree.steiner_tree(graph, terminals, weight=weight, method='kou')

    # Calculate the total cost of the Steiner tree
    total_cost = T.size(weight=weight)

    # Retrieve the selected edges in the Steiner tree
    selected_edges_scenario = list(T.edges())

    return total_cost, selected_edges_scenario

def add_relevance_scores(ssfp: SSFP, score_dict: Dict):
    """
    Based on the score_dict obtained from the wait-and-see approach, we compute the relevancy score per edge.
    :param ssfp: SSFP object.
    :param score_dict: dictionary which stores which edge is used in which scenario / subproblem.
    :return: SSFP object with a Networkx graph containing relevancy scores for each edge as an attribute.
    """
    # We calculate the score for each edge, weighted by the probability of the scenario.
    relevancy_scores = {}
    for edge in ssfp.graph.edges():
        relevancy_scores[edge] = sum(score_dict[scenario.id][frozenset(edge)] * scenario.probability for scenario in ssfp.future)

    # We normalize the scores
    c_max = max(relevancy_scores.values())

    # For all edges in relevancy_scores, we calculate the score c_max - score.
    step2_score_dict = {edge: c_max - score for edge, score in relevancy_scores.items()}

    # We add the relevancy scores as an attribute to the SSFP graph.
    for edge in ssfp.graph.edges():
        ssfp.graph.edges[edge]["relevance scores"] = step2_score_dict[edge]

    return ssfp


def get_second_stage_edge_costs(ssfp: SSFP):
    """
    Gets the second stage edge costs for each edge and returns them in a dictionary.
    :param ssfp: SSFP object.
    :return: dictionary of edge costs.
    """
    edge_costs = {}
    for scenario in ssfp.future:
        edge_costs[scenario.id] = {}

        for edge in ssfp.graph.edges():
            edge_costs[scenario.id][edge] = ssfp.graph.edges[edge][f"weight second stage scenario {scenario.id} pipe 1"]

    return edge_costs


def reset_second_stage_edge_costs(ssfp: SSFP, edge_costs: Dict):
    """
    Resets the second stage edge costs in the SSFP object.
    :param ssfp: SSFP object.
    :param edge_costs: dictionary of second stage edge costs.
    :return: SSFP object with updated second stage edge costs.
    """
    for scenario in ssfp.future:
        for edge in ssfp.graph.edges():
            ssfp.graph.edges[edge][f"weight second stage scenario {scenario.id} pipe 1"] = edge_costs[scenario.id][edge]

    return ssfp


def compute_costs_on_subset(ssfp: SSFP, first_stage_selected_edges: List, subset_ratio: float,
                            node_to_center: Dict = None, extra_runs: bool = False, first_stage=False, exact=False,
                            node_files=""):
    """
    We compute the costs of a solution (with first stage edges) with a subset of the original scenario tree.
    :param ssfp: SSFP object.
    :param first_stage_selected_edges: list of edges that are selected in the first stage.
    :param subset_ratio: ratio of the sample size of the original scenario tree.
    :return: objective score of the SSTP-solution.
    """

    # Take a subset of the scenarios.
    subset_scenarios = take_subset(ssfp, subset_ratio)

    # Store the original edge costs of the SSFP
    original_edge_costs = get_second_stage_edge_costs(ssfp)

    # Initialize the parameters
    # Objective of the optimization problem
    objective = 0

    # Initialize the score_dict variable: stores if an edge is selected in a scenario
    score_dict = {}

    # Take a deepcopy of the ssfp object
    dummy_ssfp = copy.deepcopy(ssfp)

    # Compute size of the subset (or sample) size.
    subset_size = math.ceil(subset_ratio * len(ssfp.future))

    # Add the first stage costs to the objective
    for edge in first_stage_selected_edges:
        objective += ssfp.graph.edges[edge]["weight first stage pipe 1"]

    # Loop over the scenarios and solve the subproblems (independently of each other)
    for scenario in subset_scenarios:
        # Initialize the score_dict for the scenario
        score_dict[scenario.id] = {frozenset(e): 0 for e in ssfp.graph.edges()}

        if subset_ratio == 1:
            probability_scenario = scenario.probability
        else:
            probability_scenario = 1 / subset_size

        # Edges acquired in the first stage are free in the second stage
        for edge in first_stage_selected_edges:
            ssfp.graph.edges[edge][f"weight second stage scenario {scenario.id} pipe 1"] = 0

        original_terminals = scenario.terminal_groups
        if node_to_center:
            terminals = [node_to_center[n] for n in scenario.terminal_groups[0]]
        else:
            terminals = scenario.terminal_groups[0]
        scenario.terminal_groups = [terminals]

        if exact:
            total_cost, selected_edges_scenario = compute_exact_steiner_tree(ssfp, dummy_ssfp, scenario, f"weight second stage scenario {scenario.id} pipe 1", node_files=node_files)

        scenario.terminal_groups = original_terminals

        if not exact:
            if first_stage:
                weight = "weight first stage pipe 1"
            else:
                weight = f"weight second stage scenario {scenario.id} pipe 1"
            total_cost, selected_edges_scenario = compute_steiner_tree(ssfp.graph, terminals, weight)

        # Update the two variables accordingly
        score_dict = update_score(score_dict, scenario, selected_edges_scenario)

        objective += total_cost * probability_scenario

    # Reset the SSFP-graph object
    ssfp = reset_second_stage_edge_costs(ssfp, original_edge_costs)

    return objective, score_dict, ssfp


def reduced_cost_heuristic(ssfp: SSFP, selected_edges: List, original_objective: float, addition: bool, candidate_edges: List, score_dict: Dict):
    """
    Implementation of the reduced cost heuristic mentioned by Hokama et al. (2014), section 3.2
    :param ssfp: SSFP object.
    :param selected_edges: list of edges that are selected in the first stage.
    :param original_objective: objective of the original solution.
    :param addition: boolean indicating whether to add or remove an edge to/from the solution.
    :param candidate_edges: list of edges that are candidates for addition.
    :param score_dict: dictionary which stores which edge is used in which scenario / subproblem.
    :return: final objective of the SSTP solution and the list of the selected edges.
    """
    # Initialize the parameters
    # Boolean indicating whether the stopping criterion is met.
    stopping_criterion = False

    # Store the objective of the current solution
    current_objective = original_objective

    # We stop when no greedy improvement can be made anymore.
    while not stopping_criterion:
        # We store the improvement counter as it should trigger the stopping criterion (when it's equal to zero).
        improvement_counter = 0

        # We loop over the edges and check if adding an edge yields an improvement over the original objective.
        for edge in candidate_edges:
            # We make a candidate solution and compute its corresponding objective...
            if addition:
                candidate_solution = selected_edges + [edge]
            else:
                candidate_solution = [e for e in selected_edges if e != edge]
            candidate_objective, _, ssfp = compute_costs_on_subset(ssfp, candidate_solution, 1)

            # And check if it improves our current solution. If so, we update our parameters accordingly.
            if candidate_objective < current_objective:
                # # Extraatje...
                # candidate_objective, _, ssfp = compute_costs_on_subset(ssfp, candidate_solution, 1)

                selected_edges = candidate_solution
                current_objective = candidate_objective
                improvement_counter += 1

        # If no improvement has been found, we stop the while-loop.
        if improvement_counter == 0:
            stopping_criterion = True

    # We compute the final objective using the complete scenario tree
    objective, _, ssfp = compute_costs_on_subset(ssfp, selected_edges, 1)

    return objective, selected_edges

def tradeoff_now_or_not(ssfp: SSFP, score_dict: Dict, selected_edges: List, alpha: float):
    """
    We check if we include an edge based on a trade-off between the first and (weighted) second stage costs.
    :param ssfp: SSFP object.
    :param score_dict: dictionary which stores which edge is used in which scenario / subproblem.
    :param selected_edges: edges that are selected in the previous step.
    :param alpha: inflation factor for the second stage costs.
    :return: list of selected edges after the trade-off.
    """
    selected_edges_after_tradeoff = []

    for edge in ssfp.graph.edges():
        # Compute weighted future cost
        weighted_future_cost = sum(scenario.probability *
                                   ssfp.graph.edges()[edge][f"weight second stage scenario {scenario.id} pipe 1"] *
                                   score_dict[scenario.id][frozenset(edge)]
                                   for scenario in ssfp.future)

        costs_now = ssfp.graph.edges()[edge]["weight first stage pipe 1"]

        ssfp.graph.edges[edge]["trade off"] = weighted_future_cost / costs_now

        # We include the edge if the weighted future cost is higher than the current cost and the edge was in selected_edges.
        if edge in selected_edges and alpha * weighted_future_cost >= costs_now:
            selected_edges_after_tradeoff.append(edge)

    return selected_edges_after_tradeoff, ssfp

def get_candidate_edges(ssfp: SSFP, seed: int, selected_edges: List, subset_ratio: float = 0.1):
    """
    To lower runtimes, we reduce the set of candidate edges for addition in the reduced cost heuristic.
    We select edges based on their relevance scores in score_dict (in a random draw with weights).
    OLD: We only select edges whose first stage costs are in a certain range compared to their expected second stage costs.
    :param ssfp: SSFP object.
    :param seed: seed for random number generator.
    :param selected_edges: list of edges that are selected in the first stage.
    :param subset_ratio: ratio used to indicate how much of the edges we select as candidates.
    :return: list of candidate edges.
    """
    # Another test
    edge_cost_ratios = {e: max([ssfp.graph.edges[e][f"weight second stage scenario {s.id} pipe 1"] for s in ssfp.future])/ssfp.graph.edges[e]["weight first stage pipe 1"] for e in ssfp.graph.edges() if e not in selected_edges}
    top_count = math.ceil(len(edge_cost_ratios) * subset_ratio)
    candidate_edges = sorted(edge_cost_ratios, key=edge_cost_ratios.get, reverse=True)[:top_count]

    return candidate_edges

    # Test: just select top 20% based on relevance scores.
    # Get all edges and their scores (but discard already selected edges).
    edges_with_scores = [(u, v, data['trade off']) for u, v, data in ssfp.graph.edges(data=True) if (u, v) not in selected_edges]

    # Sort by score (ascending) and take the top 20%
    top_count = math.ceil(len(edges_with_scores) * subset_ratio)
    lowest_edges = sorted(edges_with_scores, key=lambda x: x[2], reverse=True)[:top_count]

    # Extract just the edge tuples (u, v)
    candidate_edges = [(u, v) for u, v, _ in lowest_edges]

    return candidate_edges

def correct_ssfp(ssfp: SSFP, pre_selected_edges: List):
    """
    Loop over all the edge in pre_selected_edges and set their second stage costs equal to zero.
    :param ssfp: SSFP object.
    :param pre_selected_edges: list of edges that are already selected for the first stage.
    :return: corrected SSFP object.
    """
    for scenario in ssfp.future:
        for edge in pre_selected_edges:
            ssfp.graph.edges()[edge][f"weight second stage scenario {scenario.id} pipe 1"] = 0

    return ssfp

def aggregate_nodes(graph: nx.Graph):
    """
    Cluster nodes that lie close to each other.
    :param graph: Networkx graph.
    :return: dictionary mapping each node to a cluster.
    """
    # Step 1: Compute the all-pairs shortest path distance matrix
    nodes = list(graph.nodes())
    n = len(nodes)
    dist_matrix = np.zeros((n, n))

    for i, u in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(graph, u, weight='weight first stage pipe 1')
        for j, v in enumerate(nodes):
            dist_matrix[i, j] = lengths.get(v, np.inf)  # inf if no path

    # Step 2: Run DBSCAN with a precomputed distance matrix
    # Set appropriate eps (maximum distance) and min_samples
    eps = dist_matrix.max() * 0.50
    dbscan = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
    labels = dbscan.fit_predict(dist_matrix)

    # Step 3: For each cluster, find the node with smallest total distance to other cluster members
    cluster_to_center = {}
    for label in set(labels):
        if label == -1: continue  # skip noise
        cluster_indices = [i for i, l in enumerate(labels) if l == label]
        submatrix = dist_matrix[np.ix_(cluster_indices, cluster_indices)]

        total_dists = np.sum(submatrix, axis=1)
        center_idx = cluster_indices[np.argmin(total_dists)]
        center_node = nodes[center_idx]
        cluster_to_center[label] = center_node

    # Step 4: Map each node to the center of its cluster
    node_to_center = {
        nodes[i]: cluster_to_center[labels[i]]  if labels[i] != -1 else n
        for i, n in enumerate(nodes)
    }

    return node_to_center


def heuristic3(ssfp: SSFP, sample_size: int, exact: bool, nodefiles: str):
    """
    This heuristic is based on a conference paper.
    :param ssfp: original SSFP object.
    :param sample_size: size of each sample.
    :return: objective runtime, and solution of the algorithm.
    """
    # Trick for feasible sample_size
    sample_size = min(sample_size, len(ssfp.future))

    # We start measuring the time
    start_time = time.time()

    # Dummy object, used for later
    dummy_ssfp = copy.deepcopy(ssfp)
    dummy_graph = copy.deepcopy(ssfp.graph)

    # Binary parameter y^s_e equals 1 if we use edge e in scenario s, and 0 otherwise
    y = {s.id: {frozenset(e): 0 for e in ssfp.graph.edges()} for s in ssfp.future}

    # Loop over the future scenarios (in groups of size 'sample_size').
    groups = [ssfp.future[i:i+2] for i in range(0, len(ssfp.future), sample_size)]

    # Loop over the groups, collect the union of terminals, and solve the corresponding STP
    for group in groups:
        if sample_size > 1:
            all_terminals = set()

            for scenario in group:
                all_terminals |= set(scenario.terminal_groups[0])

            # Within the group_selected_edges, determine which edge is used for which scenario.
            if exact:
                ssfp.present.terminal_groups = [list(all_terminals)]
                _, subset_edges = compute_exact_steiner_tree(ssfp, dummy_ssfp, ssfp.present, 'weight first stage pipe 1',
                                                             nodefiles)

                subset_graph = copy.deepcopy(ssfp.graph)
                edges_to_remove = [e for e in subset_graph.edges if e not in subset_edges and (e[1], e[0]) not in subset_edges]
                subset_graph.remove_edges_from(edges_to_remove)
                ssfp.graph = subset_graph
            else:
                subset_graph = steinertree.steiner_tree(ssfp.graph, all_terminals,
                                                        weight='weight first stage pipe 1', method='kou')
        else:
            subset_graph = ssfp.graph

        for scenario in group:
            weight = f'weight second stage scenario {scenario.id} pipe 1'
            if exact:
                _, scenario_selected_edges = compute_exact_steiner_tree(ssfp, dummy_ssfp, scenario, weight, nodefiles)
            else:
                _, scenario_selected_edges = compute_steiner_tree(subset_graph, scenario.terminal_groups[0],
                                                               weight=weight)

            ssfp.graph = dummy_graph

            # Update the parameter y^s_e
            for e in scenario_selected_edges:
                y[scenario.id][frozenset(e)] = 1

    selected_edges_after_tradeoff, ssfp = tradeoff_now_or_not(ssfp, y, ssfp.graph.edges(), 1)

    # Evaluate
    objective, _, _ = compute_costs_on_subset(ssfp, selected_edges_after_tradeoff, 1, exact=exact)

    # We stop measuring the time
    end_time = time.time()
    runtime = end_time - start_time

    return objective, runtime, selected_edges_after_tradeoff


def wait_and_see(ssfp: SSFP, csv_file: str, node_files: str, exact_solver: bool):
    """
    This function tests the approach of wait-and-se, so install nothing in the first stage and everything in the
    second stage.
    :param ssfp: SSFP object.
    :param csv_file: name of csv file in which we store the runtime and objective.
    """
    # Start measuring the time
    start_time = time.time()

    objective, _, ssfp = compute_costs_on_subset(ssfp, [], 1, exact=exact_solver, node_files=node_files)

    # Stop measuring the time
    end_time = time.time()
    runtime = end_time - start_time

    # Store the results in a csv file
    with open(csv_file, "a") as f:
        print(f"{len(ssfp.future)};{runtime};{objective}", file=f)
        f.flush()


def main(basefile: str, instance_folder: str, number_of_scenarios: List[int], run_lsde: bool, run_heuristic1: bool,
         ws_benchmark: bool, print_info: bool, nodefiles: str, max_runtime: int, original_objective: bool, alpha: float,
         loop: bool, reduced_cost: bool, first_step: bool, seed: int, run_heuristic3: bool, exact_solver: bool,
         sample_size: int):
    # If not already existing, make subfolders for the paths
    paths = [f"heuristic/wait_and_see_seed_{seed}_exact_solver_{exact_solver}", f"heuristic/exact_solver_{exact_solver}_sample_size_{sample_size}"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    # Create csv files where we store basic information from the experiments.
    # LSDE_v1
    if run_heuristic3:
        folder = f"heuristic/exact_solver_{exact_solver}_sample_size_{sample_size}"
        csv_file = f"{folder}/result_{basefile}.csv"
        with open(csv_file, "w") as f:
            print("Number of scenarios;Runtime;Objective", file=f)
            f.flush()
    elif ws_benchmark:
        folder = f"heuristic/wait_and_see_seed_{seed}_exact_solver_{exact_solver}"
        csv_file = f"{folder}/result_{basefile}.csv"
        with open(csv_file, "w") as f:
            print("Number of scenarios;Runtime;Objective", file=f)
            f.flush()

    # Iterate over the number of scenarios
    for s in number_of_scenarios:
        file = f"{basefile}-{s}s"
        if "VIENNA" in instance_folder:
            extension = "sstp"
        else:
            extension = "stp"
        path = f"dimacs_data/{instance_folder}/{file}.{extension}"

        ssfp = bd.make_ssfp(path)

        # We print some basic information about the instance
        if print_info:
            print("Number of nodes", len(ssfp.graph.nodes()))
            print("Number of edges", len(ssfp.graph.edges()))
            print("Number of scenarios", len(ssfp.future))

        # Turn the SSFP into an SSTP
        ssfp = ssfp_to_sstp(ssfp)

        if ws_benchmark:
            wait_and_see(ssfp, csv_file, nodefiles, exact_solver)
        else:
            # heuristic2(ssfp, max_runtime, folder, file, print_info, nodefiles, original_objective)
            objective, runtime, solution = heuristic3(ssfp, sample_size, exact_solver, nodefiles)

            # We store the solution for the first pipe in a file as a .txt file
            with open(f"{folder}/solution_{file}.txt", "w") as f:
                print(solution, file=f)

            # We store the results in a csv file
            with open(csv_file, "a") as f:
                print(f"{len(ssfp.future)};{runtime};{objective}", file=f)
                f.flush()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments for the SSFP")
    parser.add_argument("--instance_folder", type=str, default="SSTP-LINS", help="The folder with the instances")
    parser.add_argument("--number_of_scenarios", type=int, nargs="+", default=[5, 10, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000], help="The number of scenarios")
    parser.add_argument("--basefile", type=str, default="lin01", help="The basefile")
    parser.add_argument("--run_lsde", action="store_true", help="Run the LSDE_v1 algorithm")
    parser.add_argument("--run_heuristic1", action="store_true", help="Run the first heuristic")
    parser.add_argument("--ws_benchmark", action="store_true", help="Run the wait-and-see benchmark")
    parser.add_argument("--print_info", action="store_true", help="Print basic information")
    parser.add_argument("--nodefiles", type=str, default="", help="The nodefiles")
    parser.add_argument("--max_runtime", type=int, default=3600, help="The maximum runtime")
    parser.add_argument("--original_objective", action="store_true", help="Compute the objective of the original problem")
    parser.add_argument("--alpha", type=float, default=1.0, help="The alpha parameter for heuristic 1 (default is 1)")
    parser.add_argument("--loop", action="store_true", help="Run loop.")
    parser.add_argument("--reduced_cost", action="store_true", help="Run reduced cost heuristic.")
    parser.add_argument("--first_step", action="store_true", help="Run the first step.")
    parser.add_argument("--seed", type=int, default=1, help="The seed of the random number generator.")
    parser.add_argument("--run_heuristic3", action="store_true", help="Run heuristic3.")
    parser.add_argument("--exact_solver", action="store_true", help="Use exact solver for STP.")
    parser.add_argument("--sample_size", type=int, default=1, help="The sample size of the heuristic.")

    main(**vars(parser.parse_args()))
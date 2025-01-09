import networkx as nx
from src.objects import *
import argparse
import src.stochastic as so
import src.deterministic as do
import src.scenario_selection as sc
from gurobipy import GRB
import time
import math
import random
random.seed(98)


def read_graph(path, ssfp_instance=True):
    """
    Read a graph from a file in the DIMACS format.
    """
    with open(path, "r") as file:
        lines = file.readlines()

    graph = nx.Graph()

    edges = {}
    terminals_per_scenario = {}
    edge_counter = 0

    for line in lines:
        if line.startswith("E "):
            _, u, v, w = line.split()
            edges[(int(u), int(v))] = {"weight first stage pipe 1": float(w)}

        if line.startswith("SP"):
            probabilities = line.split()[1:]
            probabilities = [float(p) for p in probabilities]

        if line.startswith("SE "):
            u, v = list(edges.keys())[edge_counter]
            weights = line.split()[1:]
            for s, weight in enumerate(weights):
                edges[(int(u), int(v))][f"weight second stage scenario {s+1} pipe 1"] = float(weight)
            edge_counter += 1

        if line.startswith("ST"):
            node = int(line.split()[1])
            terminals = line.split()[2:]
            for s in range(len(terminals)):
                if int(terminals[s]) == 1:
                    if s not in terminals_per_scenario.keys():
                        terminals_per_scenario[s] = list()
                    terminals_per_scenario[s].append(node)

    for (u, v) in edges.keys():
        graph.add_edge(int(u), int(v))
        for k, weight in edges[(u, v)].items():
            graph.edges[(u, v)][k] = weight

    if ssfp_instance:
        for s, terminals in terminals_per_scenario.items():
            candidates = list(set(graph.nodes()) - set(terminals))
            disjoint_subsets = get_disjoint_subsets(candidates, 5, 2)
            disjoint_subsets.append(terminals)
            terminals_per_scenario[s] = disjoint_subsets

    return graph, terminals_per_scenario, probabilities


def read_vienna(path, ssfp_instance=True):
    """
    Read a graph from a file in the Vienna format.
    """
    with open(path, "r") as file:
        lines = file.readlines()

    terminals_per_scenario = {}
    node_reader = False
    edge_reader = False

    edges = {}

    for i, line in enumerate(lines):
        if lines[i-1].startswith("# nr scenarios"):
            number_of_scenarios = int(line.split()[0])
        elif lines[i-1].startswith("probabilities"):
            probabilities = [float(p) for p in line.split()]
        elif line.startswith("#id"):
            node_reader = True
            continue
        elif line.startswith("link"):
            node_reader = False
            edge_reader = True
            continue
        elif node_reader:
            node = line.split()[0]
            node_info = line.split()[-number_of_scenarios:]
            for s in range(number_of_scenarios):
                if int(node_info[s]) == 1:
                    if s not in terminals_per_scenario.keys():
                        terminals_per_scenario[s] = list()
                    terminals_per_scenario[s].append(node)
        elif edge_reader:
            edge_info = line.split()[1:]
            u = edge_info[0]
            v = edge_info[1]

            edges[(int(u), int(v))] = {"weight first stage pipe 1": float(edge_info[2])}

            for s in range(number_of_scenarios):
                edges[(int(u), int(v))] = {f"weight second stage {s+1} pipe 1": float(edge_info[3+s])}

    graph = nx.Graph()
    for (u, v) in edges.keys():
        graph.add_edge(int(u), int(v))
        for k, weight in edges[(u, v)].items():
            graph.edges[(u, v)][k] = weight

    if ssfp_instance:
        for s in range(number_of_scenarios):
            terminals = terminals_per_scenario[s]
            candidates = list(set(graph.nodes()) - set(terminals))
            disjoint_subsets = get_disjoint_subsets(candidates, 5, 5)
            disjoint_subsets.append(terminals)
            terminals_per_scenario[s] = disjoint_subsets

    return graph, terminals_per_scenario, probabilities


def random_subset(lst):
    # Determine the size of the subset (random size between 0 and the length of the list)
    subset_size = random.randint(1, len(lst))

    # Return a random subset of the original list
    return random.sample(lst, subset_size)


def get_disjoint_subsets(large_list, subset_size, num_subsets):
    # Shuffle the large list randomly
    random.shuffle(large_list)

    disjoint_subsets = []
    i = 0
    while len(disjoint_subsets) < num_subsets:
        subset = large_list[i:i + subset_size]
        # Check if the subset is disjoint with previous subsets
        is_disjoint = all(set(subset).isdisjoint(existing_subset) for existing_subset in disjoint_subsets)
        if is_disjoint:
            disjoint_subsets.append(subset)
        i += subset_size

    return disjoint_subsets

def make_ssfp(path):
    """
    Create an SSFP instance from a file.
    :param path: file describing the deterministic instance.
    :return: SSFP-object.
    """
    if "VIENNA" in path:
        graph, terminals_per_scenario, probabilities = read_vienna(path)
    else:
        graph, terminals_per_scenario, probabilities = read_graph(path)

    # Generating the pipes
    pipe1 = Pipe("Pipe 1", 1)
    pipe2 = Pipe("Pipe 2", 2)
    all_pipes = [pipe1, pipe2]

    # Making the scenarios
    present = Scenario(terminals_per_scenario[0], None, list(graph.edges()), random_subset(all_pipes), "Present", 0)
    # present = Scenario([[]], probabilities[0], list(graph.edges()), [pipe], "Present", 0)
    future = [Scenario(terminals_per_scenario[s], probabilities[s], list(graph.edges()), random_subset(all_pipes), f"Future {s+1}", s+1)
              for s in terminals_per_scenario.keys()]

    ssfp = SSFP(graph, all_pipes, present, future)

    # Set the costs of pipe 2 twice as high as the costs of pipe 1.
    eta = 2
    for edge in list(graph.edges()):
        ssfp.graph.edges()[edge][f"weight first stage pipe {pipe2.id}"] = eta * ssfp.get_first_stage_weight(edge, pipe1)
        for s in ssfp.future:
            ssfp.graph.edges()[edge][f"weight second stage scenario {s.id} pipe {pipe2.id}"] = eta * ssfp.get_second_stage_weight(edge, pipe1, s)

    ssfp.future.sort(key=lambda scenario: scenario.id, reverse=False)

    return ssfp


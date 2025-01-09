from .objects import Scenario
from typing import List, Tuple, Set, Union
import gurobipy as gp

def find_index_terminal(terminal_group: List[List[int]], terminal: int) -> int:
    """
    Finds the index of the terminal in the terminal group.
    :param terminal_group: nested list of terminals.
    :param terminal: terminal of which the terminal group index has to be found.
    :return: index of the terminal group.
    """
    for i, group in enumerate(terminal_group):
        if terminal in group:
            return i


def edges_to_arcs(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Transforms list of edges to list of arcs.
    :param edges: list of edges.
    :return: list of arcs.
    """
    edges_opposite = [(v, u) for u, v in edges]
    return edges + edges_opposite


def get_terminals(terminal_group: List[List[int]]) -> List[int]:
    """
    Turns a nested list of terminals into a list of terminals.
    :param terminal_group: nested list of terminals.
    :return: list of terminals.
    """
    return [t for group in terminal_group for t in group]


def get_terminal_groups_until_k(
    terminal_group: List[List[int]], group_index: int
) -> Set[int]:
    """
    Get terminal groups until index k.
    :param terminal_group: nested list of terminals.
    :param group_index: index of the terminal group.
    :return: subset of terminal groups up till index k.
    """
    return set(get_terminals(terminal_group[:group_index]))


def terminal_groups_without_root(
    terminal_group: List[List[int]], roots: List[int], group_index: int
) -> Set[int]:
    """
    Get terminal groups until index k without kth root.
    :param terminal_group: nested list of terminals.
    :param roots: list of roots.
    :param group_index: index of the terminal group.
    :return: subset of terminal groups from index k to K.
    """
    if len(terminal_group[0]) > 0:
        return set(get_terminals(terminal_group[group_index:])) - set([roots[group_index]])
    else:
        return set()


def deduct_sets(
    scenario: Scenario,
) -> Tuple[List[Tuple[int, int]], List[int], List[int], List[int], List[int]]:
    """
    Deducts sets from other sets.
    :param scenario: Scenario-object.
    :return: list of (allowed) arcs, list of terminals, list of roots, list of vertices.
    """
    # Get arcs from edges
    arcs = edges_to_arcs(scenario.edges)
    terminals = [terminal for group in scenario.terminal_groups for terminal in group]

    # Included this if-statement for the case that there are no terminal groups in the first stage.
    if len(scenario.terminal_groups[0]) > 0:
        roots = [group[0] for group in scenario.terminal_groups]
    else:
        roots = []

    vertices = list(scenario.parent.graph.nodes())
    k_list = range(len(scenario.terminal_groups))

    all_pipes = [p.id for p in scenario.parent.all_pipes]
    pipes = [p.id for p in scenario.pipes]

    return arcs, terminals, roots, vertices, k_list, all_pipes, pipes


def demand_and_supply_undirected(v: int, t: int, r: int) -> int:
    """
    Determines where a vertex is a root, terminal, or neither.
    :param v: vertex.
    :param t: terminal.
    :param r: root.
    :return: 1 if vertex is the root, -1 if vertex is a terminal, and 0 otherwise.
    """
    if v == r:
        return 1
    elif v == t:
        return -1
    else:
        return 0


def demand_and_supply_directed(
    model: gp.Model, scenario: Scenario, t: int, v: int, roots: List[int], k: int
) -> Union[gp.Var, int]:
    """
    Similar to demand_and_supply_undirected, but for the directed models.
    :param z: decision variable z.
    :param terminal_groups: nested list of terminals.
    :param t: terminal.
    :param v: vertex.
    :param roots: list of roots.
    :param k: index of the terminal group.
    :return: z_{kl} if vertex is the root, -z_{kl} if vertex is a terminal, and 0 otherwise.
    """
    variable_name = (
        f"{scenario.id}_z[{k},{find_index_terminal(scenario.terminal_groups, t)}]"
    )
    z = model.getVarByName(variable_name)
    if v == roots[k]:
        return z
    elif v == t:
        return -z
    else:
        return 0

import networkx as nx
from typing import List, Tuple, Callable
import gurobipy as gp


class SSFP:
    """
    This object represents the 2S-SSFP as described in (Markhorst et al., 2025).
    """
    def __init__(self, graph: nx.Graph, all_pipes: List['Pipe'], present: 'Scenario', future: List['Scenario']):
        """
        :param graph: networkx object.
        :param all_pipes: set of all pipes that are considered in the SSFP.
        :param present: Scenario object that represents the present scenario.
        :param future: list of Scenario objects that represent the future scenarios.
        """
        self.graph = graph
        self.all_pipes = all_pipes
        self.all_edges = list(graph.edges())
        self.present = present
        self.future = future
        self.connect_scenarios_to_ssfp()

    def connect_scenarios_to_ssfp(self):
        """
        Connects the scenarios to the SSFP.
        """
        self.present.set_parent(self)
        for scenario in self.future:
            scenario.set_parent(self)

    def scenario_idx(self, scenario: 'Scenario') -> int:
        """
        Returns the index of the scenario in the future list.
        """
        return self.future.index(scenario)

    def __repr__(self) -> str:
        return f"SSFP graph with {len(self.graph.nodes())} vertices and {len(self.graph.edges())} edges"

    def get_first_stage_weight(self, edge: Tuple[int, int], pipe: 'Pipe') -> float:
        """
        Returns the first stage weight of the edge for the given pipe.
        """
        return self.graph.edges()[edge][f"weight first stage pipe {pipe.id}"]

    def get_second_stage_weight(self, edge: Tuple[int, int], pipe: 'Pipe', scenario: 'Scenario') -> float:
        """
        Returns the second stage weight of the edge for the given pipe.
        """
        return self.graph.edges()[edge][f"weight second stage scenario {scenario.id} pipe {pipe.id}"]

    def set_pipe_costs(self, eta: int, pipe: 'Pipe'):
        """
        Set the first stage costs of the pipe to eta times the first stage costs of the first pipe.
        """
        for edge in list(self.graph.edges()):
            self.graph.edges()[edge][f"weight first stage pipe {pipe.id}"] = eta * self.get_first_stage_weight(edge, self.all_pipes[0])

    def set_second_stage_costs(self, cost_increase_method: Callable, pipe: 'Pipe'):
        """
        Set the second stage costs of the pipe to the first stage costs of the pipe multiplied with a factor determined
        by the cost_increase_method.
        :param cost_increase_method: method that computes the increase rate per edge.
        :param pipe: pipe for which the costs are set.
        """
        for edge in list(self.graph.edges()):
            for scenario in self.future:
                self.graph.edges()[edge][f"weight second stage scenario {scenario.id} pipe {pipe.id}"] = cost_increase_method(self.graph, edge) * \
                                                                          self.get_first_stage_weight(edge, pipe)


class Scenario:
    """
    This object represents a scenario within the 2S-SSFP as described in (Markhorst et al., 2025).
    """
    def __init__(self, terminal_groups: List[List[int]], probability: float, edges: List[Tuple[int]], pipes: List['Pipe'], name: str, id: int):
        """
        :param terminal_groups: list of terminal groups that are considered in the SSFP.
        :param probability: probability of this scenario taking place.
        :param edges: list of the edges that can be used to install pipes on.
        :param pipes: list of the pipes that can be used to route through.
        :param name: name of the scenario.
        :param id: unique id of the scenario.
        """
        self.terminal_groups = terminal_groups
        self.probability = probability
        self.edges = edges
        self.arcs = edges + [(v, u) for (u, v) in edges]
        self.pipes = pipes
        self.name = name
        self.parent = None
        self.id = id
        self.digraph = None

    def set_parent(self, parent: SSFP):
        self.parent = parent

    def __repr__(self):
        return f"Scenario {self.name}"

    def add_digraph(self):
        self.digraph = self.parent.graph.to_directed()



class Solution:
    """
    This object represents the solution for an 2S-SSFP as described in (Markhorst et al., 2023).
    """
    def __init__(self, model: gp.Model, ssfp: SSFP, compilation_time: float, name: str):
        self.model = model
        self.ssfp = ssfp
        self.compilation_time = compilation_time
        self.run_time = model.Runtime
        self.name = name
        self.route = self.get_results()
        self.first_stage_costs = self.get_first_stage_costs()
        self.objective = model.ObjVal

    def __repr__(self):
        return f"Solution from {self.name}"

    def get_first_stage_costs(self):
        """
        Returns the first stage costs of the solution.
        """
        if self.model.Status == 2:
            return sum(self.model.getVarByName(f"{self.ssfp.present.id}_x[{p.id},{u},{v}]").X * self.ssfp.graph.edges[(u, v)][f"weight first stage pipe {p.id}"]
                       for (u, v) in list(self.ssfp.graph.edges())
                       for p in self.ssfp.all_pipes)
        else:
            return None

    def get_second_stage_costs(self):
        """
        Returns the second stage costs of the solution.
        """
        return sum((self.model.getVarByName(f"{scenario.id}_x[{p.id},{u},{v}]").X - self.model.getVarByName(f"{self.ssfp.present.id}_x[{p.id},{u},{v}]").X)* self.ssfp.graph.edges[(u, v)][f"weight second stage scenario {scenario.id} pipe {p.id}"] * scenario.probability
                   for (u, v) in list(self.ssfp.graph.edges())
                   for p in self.ssfp.all_pipes
                   for scenario in self.ssfp.future)

    def get_results(self):
        """
        :return: nested dictionary with 1 if pipe p uses edge e and 0 otherwise.
        """
        if self.model.Status == 2:
            first_stage_result = {p: {(u, v): round(self.model.getVarByName(f"{self.ssfp.present.id}_x[{p.id},{u},{v}]").X)
                                      for (u, v) in list(self.ssfp.graph.edges())}
                                  for p in self.ssfp.all_pipes}
            return first_stage_result
        else:
            return None


class Pipe:
    """
    This object represents a pipe-type.
    """
    def __init__(self, name: str, id: int):
        """
        :param name: name of the pipe type.
        :param id: id of the pipe type.
        """
        self.name = name

        # ID counter starts at 1
        self.id = id

    def __repr__(self):
        return f"Pipe {self.id}"






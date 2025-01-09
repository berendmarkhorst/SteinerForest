import gurobipy as gp
import networkx as nx
from gurobipy import GRB
from itertools import combinations
import numpy as np
import copy
import time
import math

eps = 1e-6

class Scenario():
    """
    Scneario class to store the demand vector and the probability of a scenario.
    """
    def __init__(self, id, demand_vector, probability):
        self.demand = demand_vector
        self.id = id
        self.probability = probability

class CTSP():
    """
    Class to store the CTSP instance.
    """
    def __init__(self, graph, demand_matrix, capacity, probabilities):
        self.scenarios = [Scenario(id, demand_vector, probabilities[id]) for id, demand_vector in enumerate(demand_matrix)]
        self.graph = graph.to_directed()
        self.capacity = capacity

    def add_constraints(self, model, scenario, x):
        """
        Adds the constraints to the model for a given scenario.
        :param model: Gurobi model.
        :param scenario: Scenario object.
        :param x: Gurobi variable.
        :return: Gurobi model and variable(s).
        """
        y = model.addVars([scenario.id], self.graph.nodes(), self.graph.nodes(), vtype=GRB.BINARY, name="y")

        for i in self.graph.nodes():
            if i != 0:
                model.addConstr(gp.quicksum(y[scenario.id, i, j]
                                            for j in self.graph.nodes() if i != j) == 1)
                model.addConstr(gp.quicksum(y[scenario.id, j, i]
                                            for j in self.graph.nodes() if i != j) == 1)
                model.addConstr(y[scenario.id, i, i] == 0)
            else:
                model.addConstr(gp.quicksum(y[scenario.id, i, j]
                                            for j in self.graph.nodes() if i != j) >= 1)
                model.addConstr(gp.quicksum(y[scenario.id, j, i]
                                            for j in self.graph.nodes() if i != j) >= 1)

        for i in self.graph.nodes():
            for j in self.graph.nodes():
                if i != 0 and j != 0:
                    model.addConstr(y[scenario.id, i, j] <= x[i,j])

        model.update()

        return model, y

    def update_objective(self, model, variables):
        """
        Sets the objective of the Gurobi model.
        :param model: Gurobi model.
        :param variables: Gurobi variables.
        :return: Gurobi model.
        """
        # Objective function
        expr = gp.quicksum(scenario.probability * gp.quicksum(self.graph.edges()[(i, j)]["distance"] * variables[scenario.id][scenario.id, i,j] for (i, j) in self.graph.edges()) for scenario in self.scenarios)
        model.setObjective(expr, GRB.MINIMIZE)
        model.update()
        return model

    def solve_deterministic_equivalent(self, logfile = None, aggregation_time=0, rootnode=False, node_file=""):
        """
        Solves the deterministic equivalent of the CTSP.
        :param logfile: log file for the Gurobi model.
        :param aggregation_time: time it take to aggregate.
        :param rootnode: boolean if we only solve the root node.
        :param node_file: directory for the node files.
        :return: Gurobi model, compilation time and Gurobi variable.
        """
        # Measure the compilation time
        start = time.time()

        model = gp.Model("CTSP")
        model._ctsp = self
        model._rootnode = rootnode

        # Stop printing the log
        model.setParam("LogToConsole", 0)

        # Set the runtime to 7200 seconds
        model.setParam('TimeLimit', 7200 - aggregation_time)

        if node_file != "":
            model.setParam("NodeFileStart", 0.5)
            model.setParam("NodeFileDir", node_file)

        # Clear the logfile and start logging
        if logfile:
            with open(logfile, "w") as _:
                pass
            model.setParam("LogFile", logfile)

        # Decision variables
        x = model.addVars(self.graph.nodes(), self.graph.nodes(), vtype=GRB.BINARY, name="x")

        # First stage constraints
        for i in self.graph.nodes():
            if i != 0:
                model.addConstr(gp.quicksum(x[i, j] for j in self.graph.nodes() if i != j) == 1)
                model.addConstr(gp.quicksum(x[j, i] for j in self.graph.nodes() if i != j) == 1)
            else:
                model.addConstr(gp.quicksum(x[i, j] for j in self.graph.nodes() if i != j) >= 1)
                model.addConstr(gp.quicksum(x[j, i] for j in self.graph.nodes() if i != j) >= 1)

        model.update()

        model._cuts = []
        model._cuts_LHS = []
        model._cuts_RHS = []

        variables = {}
        for scenario in self.scenarios:
            model, variables[scenario.id] = self.add_constraints(model, scenario, x)

        model = self.update_objective(model, variables)

        model.params.Threads = 1
        model.params.LazyConstraints = 1

        model._x = x
        model._variables = variables

        # Measure the compilation time
        end = time.time()
        compilation_time = end - start

        model.optimize(subtourelim)

        return model, compilation_time, x


    def distance(self, scenario1: Scenario, scenario2: Scenario):
        """
        Computes the distance between scenario1 and scenario2.
        :param scenario1: Scenario object.
        :param scenario2: Scenario object.
        :return: distance between the two scenarios.
        """
        # Manhattan distance between the demand vectors of two scenarios
        return np.sum(np.abs(scenario1.demand - scenario2.demand))

    def fast_forward_selection(self, sample_size):
        """
        Similar to fast forward selection for SSFP.
        :param sample_size: size of the reduced sample.
        :return: reduced list of Scenario objects.
        """
        sample = list()

        delta = {frozenset((k.id, l.id)): self.distance(k, l) for i, k in
                 enumerate(self.scenarios) for l in self.scenarios[:i + 1]}
        z = np.array([sum(k.probability * delta[frozenset((k.id, l.id))] for k in self.scenarios if k.id != l.id) for l in
                      self.scenarios])
        u = self.scenarios[np.argmin(z)]
        j_set = [scenario for scenario in self.scenarios if scenario.id != u.id]
        sample.append(u)

        for t in range(1, sample_size):
            delta = {frozenset((k.id, l.id)): np.minimum(delta[frozenset((k.id, l.id))], delta[frozenset((k.id, u.id))])
                     for i, k in enumerate(j_set) for l in j_set[:i + 1]}
            z = np.array(
                [sum(k.probability * delta[frozenset((k.id, l.id))] for k in j_set if k.id != l.id) for l in j_set])
            u = j_set[np.argmin(z)]
            sample.append(u)
            j_set.remove(u)

        return sample

    def divide_over_a(self, sample):
        """
        Divides the scenarios over the aggregated scenarios.
        :param sample: reduced list of Scenario objects.
        :return: dictionary with aggregated scenarios as keys and a list of scenario ids as values.
        """
        groups = {j.id: [] for j in sample}

        for i in self.scenarios:
            distances = [self.distance(i, j) for j in sample]
            group_index = sample[np.argmin(distances)].id
            groups[group_index].append(i.id)

        return groups

    def make_aggregated_problem(self, groups):
        """
        Aggregates the scenarios based on the groups.
        :param groups: dictionary with aggregated scenarios as keys and a list of scenario ids as values.
        :return: CTSP-object.
        """
        scenarios = []

        for index, group in groups.items():
            probability = sum(self.scenarios[id].probability for id in group)

            scenario = copy.deepcopy(self.scenarios[index])
            scenario.probability = probability

            scenarios += [scenario]

        instance = copy.deepcopy(self)
        instance.scenarios = scenarios
        instance.nr_scenarios = len(scenarios)

        return instance

    def extend_model(self, model, new_scenarios_included, wall_time_passed, x, cuts=True, values=True, rootnode=False):
        """
        Second step of the TULIP approach.
        :param model: Gurobi model.
        :param new_scenarios_included: new scenarios to be included in the model.
        :param wall_time_passed: time passed since the start of the method (i.e., the first step of TULIP).
        :param x: Gurobi variable(s).
        :param cuts: boolean indicating if we use the cuts from TULIP's first step.
        :param values: boolean indicating if we use the solution from TULIP's first step.
        :param rootnode: boolean indicating if we only solve the root node in TULIP's first step.
        :return: Gurobi model and compilation time.
        """
        # Measure the compilation time
        start = time.time()

        # We have already used some of our time for the warm start.
        model.setParam('TimeLimit', max(7200 - wall_time_passed, 0.01))

        # If you don't want to use the values from the previous model as warm start, reset the model.
        if not values:
            model.reset()

        # If there were cuts from the previous model, add them as constraints here.
        tight_cut_counter = 0
        model._cut_counter = len(model._cuts)
        if cuts:
            for idx, cut_LHS in enumerate(model._cuts_LHS):
                cut_RHS = model._cuts_RHS[idx]
                if abs(sum(model._solutions[var] for var in cut_LHS) - cut_RHS) < 1e-6:
                    model.addConstr(model._cuts[idx])
                    tight_cut_counter += 1
            model.update()

        model._tight_cut_counter = tight_cut_counter
        model._cuts = []
        model._cuts_LHS = []
        model._cuts_RHS = []

        for scenario in self.scenarios:
            if scenario.id in new_scenarios_included:
                model, model._variables[scenario.id] = self.add_constraints(model, scenario, x)

        # Set rootnode to True if you only want to solve the LP relaxation of the model
        model._rootnode = rootnode

        # Set objective original CTSP
        model = self.update_objective(model, model._variables)

        # Measure the compilation time
        end = time.time()
        compilation_time = end - start

        # Optimize model
        model.optimize(subtourelim)

        return model, compilation_time

    def solve_with_scenario_selection(self, sample_size, logfile=None, rootnode=False, node_file=""):
        """
        TULIP approach for the CTSP.
        :param sample_size: size of the reduced scenario tree.
        :param logfile: log file for the Gurobi model.
        :param rootnode: boolean indicating if we only solve the root node in TULIP's first step.
        :param node_file: directory for the node files.
        :return: Gurobi model, compilation time, and aggregation time.
        """
        start_time = time.time()
        sample = self.fast_forward_selection(sample_size)
        groups = self.divide_over_a(sample)
        ctsp_agg = self.make_aggregated_problem(groups)
        new_scenarios_included = [scenario.id for scenario in self.scenarios if scenario.id not in groups.keys()]
        end_time = time.time()
        aggregation_time = end_time - start_time

        aggregated_model, compilation_time1, x = ctsp_agg.solve_deterministic_equivalent(logfile=logfile, aggregation_time=aggregation_time, rootnode=rootnode, node_file=node_file)
        first_iteration_time = aggregated_model.Runtime

        # Check if the first iteration took too long. If so, return None.
        if time.time() - start_time > 7200:
            return None, None, None

        # print("Done with aggregated model. Now extending.")
        aggregated_model._ctsp = self
        model, compilation_time2 = self.extend_model(aggregated_model, new_scenarios_included, aggregated_model.Runtime + aggregation_time, x,
                                  cuts=True, values=True)
        model._first_iteration_time = first_iteration_time

        return model, compilation_time1 + compilation_time2, aggregation_time


def subtourelim(model, where):
    """
    Adds subtour elimination constraints dynamically to the model.
    :param model: Gurobi model.
    :param where: indicates the position in the solving process.
    """
    ctsp = model._ctsp

    # Stop after the root node.
    if model._rootnode and where == GRB.Callback.MIP and model.cbGet(GRB.Callback.MIP_NODCNT) != 0:
        model.terminate()
        return

    if where == GRB.Callback.MIPSOL or (where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
        if where == GRB.Callback.MIPSOL:
            model._solutions = {var: model.cbGetSolution(var) for var in model.getVars()}
        else:
            model._solutions = {var: model.cbGetNodeRel(var) for var in model.getVars()}

        x = model._x

        # Get the values of the x variables
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(x)
            minimum_value = 0.5
        else:
            vals = model.cbGetNodeRel(x)
            minimum_value = eps

        # Compute the arcs that are used in the solution
        used_arcs = [(i, j, {"capacity": vals[i, j]}) for i in ctsp.graph.nodes() for j in ctsp.graph.nodes()] # if vals[i, j] > minimum_value ?

        # Make a support graph
        support_graph = nx.DiGraph()
        support_graph.add_edges_from(used_arcs)

        partition_list = []
        for n in ctsp.graph.nodes():
            if n != 0:
                # Find the minimum cut
                cut_value, partition = nx.minimum_cut(support_graph, 0, n, capacity="capacity")
                partition_list.append(partition[0])

                # If necessary, add the violated SEC
                if cut_value + eps < 1:
                    # Store the left and right hand side for later!
                    S = set(partition[0])
                    left_hand_side_list = [x[i,j] for i in S for j in S if j != i]
                    right_hand_side = len(S) - 1

                    # Create the cut
                    cut = gp.quicksum(left_hand_side_list) <= right_hand_side

                    # Add the lazy cut to the model
                    model.cbLazy(cut)

                    # Store the cut for later, so we can access them in the next iteration
                    model._cuts.append(cut)
                    model._cuts_LHS.append(left_hand_side_list)
                    model._cuts_RHS.append(right_hand_side)

        for s in ctsp.scenarios:
            y = model._variables[s.id]

            # Get the values of the y variables
            if where == GRB.Callback.MIPSOL:
                y_vals = model.cbGetSolution(y)
            else:
                y_vals = model.cbGetNodeRel(y)

            if where == GRB.Callback.MIPSOL:
                # Compute the arcs that are used in the solution
                used_y_arcs = [(i, j) for i in ctsp.graph.nodes() for j in
                               ctsp.graph.nodes() if y_vals[s.id, i, j] > 0.5]

                # Check if y generates a subtour which exceeds the capacity
                cycles = subtour3(ctsp.graph.nodes(), used_y_arcs)
                for cycle in cycles:
                    demand_cycle = sum(s.demand[v] for v in cycle if v != 0)
                    if demand_cycle > ctsp.capacity:
                        cycle_temp = set(cycle) - set([0])
                        left_hand_side_list = [y[s.id, i, j] for i in cycle_temp for j in set(ctsp.graph.nodes()) - set(cycle_temp)] + [y[s.id, j, i] for i in cycle_temp for j in set(ctsp.graph.nodes()) - set(cycle_temp)]
                        right_hand_side = 2 * math.ceil(demand_cycle / ctsp.capacity)

                        cut = gp.quicksum(left_hand_side_list) >= right_hand_side

                        # Add the lazy cut to the model
                        model.cbLazy(cut)

                        # Store the cut for later, so we can access them in the next iteration
                        model._cuts.append(cut)
                        model._cuts_LHS.append(left_hand_side_list)
                        model._cuts_RHS.append(right_hand_side)
            else:
                for cycle in partition_list:
                    demand_cycle = sum(s.demand[v] for v in cycle if v != 0)

                    cycle_temp = set(cycle) - set([0]) # Bij Markus checken!

                    part1 = sum(y_vals[s.id, i, j] for i in cycle_temp for j in set(ctsp.graph.nodes()) - set(cycle_temp))
                    part2 = sum(y_vals[s.id, j, i] for i in cycle_temp for j in set(ctsp.graph.nodes()) - set(cycle_temp))

                    if part1 + part2 < 2 * math.ceil(demand_cycle / ctsp.capacity):
                        left_hand_side_list = [y[s.id, i, j] for i in cycle_temp for j in set(ctsp.graph.nodes()) - set(cycle_temp)] + [y[s.id, j, i] for i in cycle_temp for j in set(ctsp.graph.nodes()) - set(cycle_temp)]
                        right_hand_side = 2 * math.ceil(demand_cycle / ctsp.capacity)

                        cut = gp.quicksum(left_hand_side_list) >= right_hand_side

                        # Add the lazy cut to the model
                        model.cbLazy(cut)

                        # Store the cut for later, so we can access them in the next iteration
                        model._cuts.append(cut)
                        model._cuts_LHS.append(left_hand_side_list)
                        model._cuts_RHS.append(right_hand_side)


# Given a tuplelist of edges, find the shortest subtour
def subtour(ctsp, edges):
    unvisited = list(ctsp.graph.nodes())
    cycle = list(ctsp.graph.nodes()) # Dummy - guaranteed to be replaced
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(thiscycle) <= len(cycle):
            cycle = thiscycle # New shortest subtour
    return cycle

def create_cvrp_graph(input_file):
    """
    Parses the input data and creates a graph representing the CVRP problem.
    Computes the Euclidean distances between nodes or uses explicit distances if provided.
    Also stores the demand per node and the vehicle capacity.

    :param input_data: String containing the CVRP problem data.
    :return: A tuple (graph, capacity) where graph is a networkx.Graph and capacity is an integer.
    """
    # Read the data into a string
    with open(input_file, "r") as file:
        input_data = file.read()

    # Initialize variables
    nodes = {}
    demands = {}
    capacity = 0
    edge_weight_type = ""
    edge_weight_format = ""
    explicit_weights = []

    # Parse input data
    lines = input_data.strip().split('\n')
    section = None

    for line in lines:
        line = line.strip()
        if line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("EDGE_WEIGHT_FORMAT"):
            edge_weight_format = line.split(":")[1].strip()
        elif line.startswith("NODE_COORD_SECTION"):
            section = "NODE_COORD_SECTION"
            continue
        elif line.startswith("DEMAND_SECTION"):
            section = "DEMAND_SECTION"
            continue
        elif line.startswith("DEPOT_SECTION"):
            section = "DEPOT_SECTION"
            continue
        elif line.startswith("EDGE_WEIGHT_SECTION"):
            section = "EDGE_WEIGHT_SECTION"
            continue
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1].strip())
        elif section == "NODE_COORD_SECTION":
            parts = line.split()
            if len(parts) == 3:
                node = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                nodes[node] = (x, y)
        elif section == "DEMAND_SECTION":
            parts = line.split()
            if len(parts) == 2:
                node = int(parts[0])
                if edge_weight_type == "EXPLICIT":
                    nodes[node] = (0, 0)
                demand = int(parts[1])
                demands[node] = demand
        elif section == "EDGE_WEIGHT_SECTION":
            parts = list(map(float, line.split()))
            explicit_weights.extend(parts)
        elif section == "DEPOT_SECTION" and line == "-1":
            break

    # Create graph and add nodes
    G = nx.Graph()
    for node in nodes:
        G.add_node(node, x=nodes.get(node, (0, 0))[0], y=nodes.get(node, (0, 0))[1], demand=demands.get(node, 0))

    if edge_weight_type == "EXPLICIT":
        # If the EDGE_WEIGHT_TYPE is EXPLICIT, we use the provided distances
        dimension = len(nodes)
        if edge_weight_format == "LOWER_COL":
            index = 0
            # Iterate over all possible pairs of nodes
            for i in range(1, dimension + 1):
                for j in range(1, i + 1):
                    if i != j:  # Exclude self-loops
                        distance = explicit_weights[index]
                        index += 1
                        G.add_edge(i, j, distance=distance)
    else:
        # If EDGE_WEIGHT_TYPE is not EXPLICIT, compute Euclidean distances
        for node1, (x1, y1) in nodes.items():
            for node2, (x2, y2) in nodes.items():
                if node1 != node2:
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    G.add_edge(node1, node2, distance=distance)

    return G, capacity


def stochastic_tsp_lib_instance(inputfile, nr_scenarios, alpha, seed):
    """
    Generates a stochastic instance from the TSP Lib.
    :param inputfile: indicates which TSPLib we are going to make stochastic.
    :param nr_scenarios: the number of scenarios we want to include in our new, stochastic instance.
    :param alpha: the alpha parameter for the Lognormal distribution.
    :param seed: the seed for the random number generator.
    :return: CTSP-object.
    """
    np.random.seed(seed)
    graph, capacity = create_cvrp_graph(inputfile)

    # Store the original capacity and demand
    original_capacity = capacity
    deterministic_demand = np.array([graph.nodes()[node]["demand"] for node in graph.nodes()])

    # Divide the demand by the capacity because of the Lognormal distribution
    for node in graph.nodes():
        graph.nodes()[node]["demand"] = graph.nodes()[node]["demand"] / capacity

    # Reset index such that it starts at 0
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="default", label_attribute=None)

    demand_matrix = np.array([[np.random.lognormal(graph.nodes()[node]["demand"], graph.nodes()[node]["demand"] * alpha) if node != 0 else 0 for node in graph.nodes()] for _ in range(nr_scenarios)])
    probabilities = [1/nr_scenarios for _ in range(nr_scenarios)]

    # Adjust the capacity to the largest increase in demand so the instance stays a feasible problem.
    eps = 1e-6
    original_demand = np.array([[graph.nodes()[node]["demand"] + eps for node in graph.nodes()] for _ in range(nr_scenarios)])
    capacity = max(1, np.max(demand_matrix / original_demand))

    instance = CTSP(graph, demand_matrix, capacity, probabilities)
    instance.original_capacity = original_capacity
    instance.original_demand = deterministic_demand

    return instance

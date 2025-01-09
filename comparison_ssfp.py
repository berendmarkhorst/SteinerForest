import benchmark_dimacs as bd
import src.stochastic as so
import src.scenario_selection as sc
import math
import time
import os
import argparse
import pickle

"""
Compares the three different methods described in the TULIP paper with each other for the SSFP.
"""

selection_rate = 0.10


def flow(ssfp, output_file, scenario, node_file):
    start = time.time()
    gurobi_log = output_file.replace(".csv", f"_gurobi_{scenario}.txt")
    solution = so.directed_flow(ssfp, 7200, gurobi_log, node_file=node_file)
    end = time.time()
    total_time = end - start
    objective = solution.objective
    optimality_gap = solution.model.MIPGap

    with open(output_file, "a") as file:
        print(f"{len(ssfp.future)};Flow;{total_time};{objective};{optimality_gap};0;0;0;0;0;{solution.compilation_time}", file=file)
        file.flush()

    # Check if there is an optimal solution found
    if solution.model.SolCount > 0:
        # Save the Gurobi solution
        solution.model.write(output_file.replace(".csv", f"_gurobi_{scenario}.sol"))
    else:
        with open(output_file.replace(".csv", f"_gurobi_{scenario}.sol"), 'w') as f:
            print("No optimal solution found.", file=f)
            f.flush()

    return total_time, objective, optimality_gap


def cut(ssfp, output_file, scenario, node_file):
    start = time.time()
    gurobi_log = output_file.replace(".csv", f"_gurobi_{scenario}.txt")
    solution = so.directed_cut(ssfp, 7200, gurobi_log, node_file=node_file)
    end = time.time()
    total_time = end - start
    objective = solution.objective
    optimality_gap = solution.model.MIPGap

    nr_cuts = len(solution.model._cuts)

    with open(output_file, "a") as file:
        print(f"{len(ssfp.future)};Cut;{total_time};{objective};{optimality_gap};0;{nr_cuts};0;0;0;{solution.compilation_time}", file=file)
        file.flush()

    # Check if there is an optimal solution found
    if solution.model.SolCount > 0:
        # Save the Gurobi solution
        solution.model.write(output_file.replace(".csv", f"_gurobi_{scenario}.sol"))
    else:
        with open(output_file.replace(".csv", f"_gurobi_{scenario}.sol"), 'w') as f:
            print("No optimal solution found.", file=f)
            f.flush()

    return total_time, objective, optimality_gap


def warm_started_cut(ssfp, output_file, scenario, node_file):
    start = time.time()
    gurobi_log = output_file.replace(".csv", f"_gurobi_{scenario}.txt")
    number_of_scenarios = math.ceil(selection_rate * len(ssfp.future))
    solution, scenario_selection_time = sc.solve_with_scenario_selection(ssfp, number_of_scenarios, gurobi_log, node_file=node_file)
    end = time.time()
    total_time = end - start

    # If the first iteration takes over two hours, we stop the method.
    if solution is None:
        with open(output_file, "a") as file:
            print(f"First iteration took too long.", file=file)
            file.flush()
        return

    objective = solution.objective
    optimality_gap = solution.model.MIPGap
    compilation_time = solution.compilation_time

    nr_cuts_second_iteration = len(solution.model._cuts)

    with open(output_file, "a") as file:
        print(f"{len(ssfp.future)};Warm-started cut;{total_time};{objective};{optimality_gap};{solution.first_iteration_time};{solution.model._cut_counter};{solution.model._tight_cut_counter};{nr_cuts_second_iteration};{scenario_selection_time};{compilation_time}", file=file)
        file.flush()

    # Check if there is an optimal solution found
    if solution.model.SolCount > 0:
        # Save the Gurobi solution
        solution.model.write(output_file.replace(".csv", f"_gurobi_{scenario}.sol"))
    else:
        with open(output_file.replace(".csv", f"_gurobi_{scenario}.sol"), 'w') as f:
            print("No optimal solution found.", file=f)
            f.flush()

    return total_time, objective, optimality_gap


def compare(folder_name, instance_name, method, node_file):
    path = f"project2/comparison_methods/{folder_name}"
    output_file = f"{path}/{method}_{instance_name}.csv"

    os.makedirs(path, exist_ok=True)

    with open(output_file, "w") as file:
        print("Number of scenarios;Method;Runtime;Objective;Optimality gap;First iteration time;Lazy cuts;Tight lazy cuts;Cuts second step;Scenario selection time;Compilation time", file=file)
        file.flush()

    scenarios = [100, 150, 200, 250]

    for n in scenarios:
        ssfp = bd.make_ssfp(f"dimacs_data/{folder_name}/{instance_name}-{n}s.stp")

        # Save instance using Pickle
        with open(f"project2/comparison_methods/{folder_name}/{instance_name}-{n}s.pkl", "wb") as file:
            pickle.dump(ssfp, file)

        if method == "warmstart":
            warm_started_cut(ssfp, output_file, n, node_file)
        elif method == "cut":
            cut(ssfp, output_file, n, node_file)
        elif method == "flow":
            flow(ssfp, output_file, n, node_file)


def main():
    parser = argparse.ArgumentParser(prog="comparison_methods")
    parser.add_argument("--folder_name", type=str)
    parser.add_argument("--instance_name", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--node_file", type=str, default="")
    args = parser.parse_args()

    compare(args.folder_name, args.instance_name, args.method, args.node_file)


if __name__ == "__main__":
    main()
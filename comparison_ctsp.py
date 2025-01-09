import src.CTSP as ctsp
import argparse
import time
import math
import pickle

"""
Compares the three different methods described in the TULIP paper with each other for the CTSP.
"""

if __name__ == "__main__":
    # New instance generator
    parser = argparse.ArgumentParser(description='Solve the CTSP.')
    parser.add_argument('--inputfile', type=str)
    parser.add_argument('--outputfile', type=str, help='The output file.')
    parser.add_argument('--method', type=str, help='The method to use.')
    parser.add_argument('--seed', type=int, help='The seed for the random number generator.')
    parser.add_argument('--nr_scenarios', type=int)
    parser.add_argument('--warmstart_percentage', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--node_file', type=str, default="")
    args = parser.parse_args()

    warmstart_percentage = args.warmstart_percentage
    gurobi_log = args.outputfile.replace(".csv", ".txt")

    instance = ctsp.stochastic_tsp_lib_instance(f"TSP_LIB/{args.inputfile}", args.nr_scenarios, args.alpha, args.seed)

    # Save instance using Pickle
    with open(f"project2/ctsp_comparison/{args.inputfile}-{args.nr_scenarios}s-{args.alpha}alpha-{args.seed}seed.pkl", "wb") as file:
        pickle.dump(instance, file)

    # Initialize the output file
    with open(f"project2/ctsp_comparison/{args.outputfile}", 'w') as f:
        print("Number of nodes;Number of edges;Number of scenarios;Method;Time;Objective;Gap;First iteration time;Lazy cuts;Tight lazy cuts;Warm start percentage;Seed;Alpha;Cuts second step;Scenario reduction time;Compilation time", file=f)
        f.flush()

    if args.method == "cut":
        # Solve the deterministic equivalent
        start = time.time()
        logfile = f"project2/ctsp_comparison/deterministic_equivalent_gurobi_log_{gurobi_log}"
        model, compilation_time, _ = instance.solve_deterministic_equivalent(logfile=logfile, node_file=args.node_file)
        time = time.time() - start

        # Extra information to store!
        first_iteration_time = 0
        cut_counter = len(model._cuts)
        tight_cut_counter = 0

        second_step_cuts = 0

        aggregation_time = 0

    elif args.method == "warmstart":
        # Solve the scenario selection
        start = time.time()
        logfile = f"project2/ctsp_comparison/scenario_selection_gurobi_log_{gurobi_log}"
        sample_size = math.ceil(warmstart_percentage * len(instance.scenarios))
        model, compilation_time, aggregation_time = instance.solve_with_scenario_selection(sample_size, logfile=logfile, rootnode=True, node_file=args.node_file)
        time = time.time() - start

        # If the first iteration takes over two hours, we stop the method.
        if model is None:
            with open(f"project2/ctsp_comparison/{args.outputfile}", 'a') as f:
                print(f"First iteration took too long.", file=f)
                f.flush()

        # Extra information to store!
        first_iteration_time = model._first_iteration_time
        cut_counter = model._cut_counter
        tight_cut_counter = model._tight_cut_counter

        second_step_cuts = len(model._cuts)

    with open(f"project2/ctsp_comparison/{args.outputfile}", 'a') as f:
        nr_nodes = len(instance.graph.nodes())
        nr_edges = len(instance.graph.edges())
        nr_scenarios = len(instance.scenarios)
        print(f"{nr_nodes};{nr_edges};{nr_scenarios};{args.method};{time};{model.ObjVal};{model.MIPGap};{first_iteration_time};{cut_counter};{tight_cut_counter};{warmstart_percentage};{args.seed};{args.alpha};{second_step_cuts};{aggregation_time};{compilation_time}", file=f)
        f.flush()

    # Check if there is an optimal solution found
    if model.SolCount > 0:
        # Save the Gurobi solution
        model.write(f"project2/ctsp_comparison/{args.outputfile.replace('.csv', f'_gurobi_{args.method}.sol')}")
    else:
        with open(f"project2/ctsp_comparison/{args.outputfile.replace('.csv', f'_gurobi_{args.method}.sol')}", 'w') as f:
            print("No optimal solution found.", file=f)
            f.flush()
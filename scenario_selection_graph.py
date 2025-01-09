import benchmark_dimacs as bd
import src.stochastic as so
import src.scenario_selection as sc
import random
import argparse

gurobi_log = "DecompositionMethods/test.txt"

def from_a_to_objective(ssfp, a):
    groups = sc.divide_over_a(ssfp, a, alpha1=1, alpha2=0)
    ssfp_agg = sc.make_aggregated_problem(ssfp, groups)

    solution_aggregated = so.directed_cut(ssfp_agg, 7200, gurobi_log)

    solution_basis = so.directed_cut(ssfp, 7200, gurobi_log, previous_result=solution_aggregated.route)

    return solution_basis

def main(input, output, method, run_id):
    # Load the benchmark instances
    ssfp = bd.make_ssfp(input)

    # Voor nu even dubbel checken
    # ssfp.present.terminal_groups = [[]]

    outputfile = f"{output}_{method}_{run_id}.csv"

    with open(outputfile, "w") as file:
        print(f"Sample Size;Objective {method} {run_id}", file=file)
        file.flush()

    for a_final in [1, 3, 5, 8, 10, 20, 35, 50]:
        print(f"{method} {run_id} Busy with a_final = ", a_final)

        if method == "random":
            random.seed(run_id)
            a_random = random.sample(ssfp.future, a_final)
            # print(a_random)
            result_selection = from_a_to_objective(ssfp, a_random).objective
        else:
            a = sc.fast_forward_selection(ssfp, a_final, alpha1=1, alpha2=0)
            # print(a)
            result_selection = from_a_to_objective(ssfp, a).objective
        with open(outputfile, "a") as file:
            print(f"{a_final};{result_selection}", file=file)
            file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="scenario_selection_graph")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--run_id", type=int)
    args = parser.parse_args()

    main(args.input, args.output, args.method, args.run_id)

#!/bin/bash
# Set Job Requirements
#SBATCH -t 115:00:00
#SBATCH --nodes=1
#SBATCH --partition=genoa
#SBATCH --ntasks=30
#SBATCH --out=slurm/slurm-%A.out
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=b.t.markhorst@student.vu.nl

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load Gurobi/10.0.1-GCCcore-11.3.0
export GRB_LICENSE_FILE="gurobi-2.lic"

for i in `seq 29 104`; do
    srun --ntasks=1 --nodes=1 --cpus-per-task=4 python scenario_selection_graph.py $(head -$i project2/scenario_selection_graph/parameters.txt | tail -1) &
done
wait
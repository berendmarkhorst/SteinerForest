# Deterministic and Stochastic Steiner Forest
This code can be used to solve the deterministic and stochastic Steiner forest problem with Gurobi. It is used
for the following papers:

> Markhorst, B., Berkhout, J., Zocca, A., Pruyn, J., & van der Mei, R. (2025). 
> Future-proof ship pipe routing: Navigating the energy transition. 
> Ocean Engineering. 
> https://doi.org/10.1016/j.oceaneng.2024.120113

> Markhorst, B., Berkhout, J., Zocca, A., Pruyn, J., & van der Mei, R. (2024). 
> Sailing through uncertainty: ship pipe routing and the energy transition. 
> International Marine Design Conference. 
> https://doi.org/10.59490/imdc.2024.891

> Markhorst, B., Leitner, M., Berkhout, J., Zocca, A., & van der Mei, R. (2024). 
> A Two-Step Warm Start Method Used for Solving Large-Scale Stochastic Mixed-Integer Problems. 
> ArXiv preprint. 
> https://doi.org/10.48550/arXiv.2412.10098

If you use our code in your research, please consider citing (one of) these papers.

## Installation
All required packages can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Explanation of the repository
The repository contains the following files:
- ```requirements.txt```: Contains all required packages to run the code.
- ```comparison_ctsp.py```: Contains the code to compare the three different methods from the TULIP paper on the CTSP.
- ```comparison_ssfp.py```: Contains the code to compare the three different methods from the TULIP paper on the SSFP.
- ```IMDC_case_study.ipynb```: Contains the code corresponding to the IMDC paper.
- ```scenario_selection_graph.py```: Contains the code corresponding to Figure 6 in the preprint TULIP paper.
- ```benchmark_dimacs.py```: Contains the code to read the instances from the DIMACS data.

The ```src```-folder contains the following files:
- ```CTSP.py```: Contains the code related to solving the CTSP.
- ```deterministic.py```: Contains the code related to solving the deterministic SSFP.
- ```stochastic.py```: Contains the code related to solving the stochastic SSFP.
- ```robust.py```: Contains the code related to solving the robust SSFP.
- ```model_helpers.py```: Contains the helper functions used for the three aformentioned .py-files.
- ```objects.py```: Contains the code for all the objects used in this repo.
- ```scenario_selection.py```: Contains the code for the fast forward scenario selection, based on a paper from Heitsch and Römisch from 2003.

The ```results_ocean_engineering```-folder contains the results from the Ocean Engineering paper and the code to generate these results.
The ```results_tulip```-folder contains the results from the TULIP paper and the code to generate these results.
Note that we have not included the ship data and corresponding pipe route results in agreement with the commercial shipyard. 
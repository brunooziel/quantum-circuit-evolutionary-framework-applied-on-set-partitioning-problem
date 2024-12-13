# Quantum circuit evolutionary framework applied on set partitioning problem

<!-- [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/brunooziel/) -->

# <p align="center"><img src="figures\convergence_curve.png" width="980"></p>
> Figure 1: Illustration of the convergence curves of the VQE, AF-QCE, and APCD-QCE algorithms for 6 qubits and 14 qubits, respectively.

## Overview
Quantum algorithms hold significant potential for solving optimization problems. Among them, variational algorithms are particularly promising for near-term quantum devices due to their hybrid quantum-classical approach to parameter optimization. However, challenges like convergence stagnation, such as barren plateaus, often hinder their performance.

This repository introduces a quantum circuit framework with variable topology to address these challenges. The framework incorporates two approaches:

1. An **ansatz-free evolutionary** method known from literature [1].
2. A novel **pseudo-counterdiabatic evolutionary** term, inspired by counterdiabatic physics [2], tailored to the Hamiltonian structure.

Both approaches were applied to the Set Partitioning Problem in this study, as detailed in [Quantum circuit evolutionary framework applid on the set partitioning problem (12/2024)]()

### Instances_Benchmark:
We evaluated our algorithms using 35 benchmark instances from Svensson et al., tailored to the Set Partitioning Problem.

### Functions:
1. ```_utility.py:``` Contains utility functions for reading benchmark files, building Hamiltonians, and solving instances using Gurobi.
2. ```qce.py:``` Implements the Quantum Circuit Evolutionary (QCE) algorithm. To choose between AF-QCE and APCD-QCE, refer to the ```experiment_qce.ipynb``` notebook.

### Notebooks:
1. ```experiment_vqe.ipynb:``` Demonstrates the Variational Quantum Eigensolver (VQE) based on the configuration used by Cacao et al. in [3].
2. ```experiment_qce.ipynb:``` Explores the AF-QCE and APCD-QCE methods in both noiseless and noisy scenarios.

# <p align="center"><img src="figures\fig_apcd_ansatz.png" width="980"></p>
> Figure 2:  Counterdiabatic circuit structure for the APCD-QCE method, where only the term $U_{PCD}$ undergoes topology variations.

## Reference

1. Franken, L., et al. "Quantum Circuit Evolution on NISQ Devices," 2022 IEEE Congress on Evolutionary Computation (CEC), Padua, Italy, 2022, pp. 1-8. Available at: [10.1109/CEC55065.2022.9870269](https://doi.org/10.1109/CEC55065.2022.9870269).
2. Hegade, N. N., Chen, X., & Solano, E. "Digitized Counterdiabatic Quantum Optimization," Physical Review Research, vol. 4, no. 4, 2022, p. L042030. Available at: [https://link.aps.org/doi/10.1103/PhysRevResearch.4.L042030](https://link.aps.org/doi/10.1103/PhysRevResearch.4.L042030).
3. Cacao, R., Cortez, L. R. C. T., & Forner, J. et al. "The Set Partitioning Problem in a Quantum Context," Optimization Letters, vol. 18, pp. 1–17, 2024. Available at: [10.1007/s11590-023-02029-1](https://doi.org/10.1007/s11590-023-02029-1).

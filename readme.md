# Optimizing Shots Assignment in VQE Measurement
By [Linghua Zhu](https://scholar.google.com/citations?user=FGBSTlMfRP0C&hl=en), [Senwei Liang](https://leungsamwai.github.io), [Chao Yang](https://github.com/) and [Xiaosong Li](https://chem.washington.edu/people/xiaosong-li).
## Introduction
Consolidating Hamiltonian terms into cliques allows 
simultaneous measurement and reduces shots, but prior knowledge of each clique, 
like amplitudes, is very limited. To tackle this challenge, we propose two novel shot assignment strategies based 
on the standard deviation estimation to refine the convergence of VQE and reduce the 
shot requirement. These strategies address measurement challenges in two distinct 
scenarios: when shots are overallocated or underallocated.

![image](quantum.png)

## Code structure
```commandline
Optimizing measurement
│   README.md    <-- You are here
│
│   qiskit-vqe-random_assignment.py  ---> Random assignment
│   qiskit-vqe-uniform_assignment.py ---> Uniform assignment
│   qiskit-vqe-variance_minimized.py ---> VMSA
│   qiskit-vqe-variance_preserved.py ---> VRSR
│   
│   readresult.ipynb ---> Result visualization
```
## How to run the code
There are three args: 
```commandline
--trial # the experiment number, which uses to name the result
--shots # Total number of shots for each iteration
--std_shots # Number of shots used to estimate the standard deviation for each clique
```
### Baseline strategies:
We first introduce two baseline strategies:
#### Uniform assignment:
```commandline
python qiskit-vqe-uniform_assignment.py --trial 1 --shots 600
```
### Our strategies:
#### Variance-Minimized Shot Assignment:
```commandline
python qiskit-vqe-variance_minimized.py --trial 1 --shots 600 --std_shots 50
```
#### Variance-Preserved Shot Reduction:
```commandline
python qiskit-vqe-variance_preserved.py --trial 1 --shots 600 --std_shots 50
```

## Citing this paper
If you find this paper helps you in your research, please kindly cite:
```
@article{zhu2023optimizing,
  title={Optimizing Shot Assignment in Variational Quantum Eigensolver Measurement},
  author={Zhu, Linghua and Liang, Senwei and Yang, Chao and Li, Xiaosong},
  journal={arXiv preprint arXiv:2307.06504},
  year={2023}
}
```

# Genetic Algorithm for Constrained Knapsack Problem

## Project Overview

This project implements a **Genetic Algorithm (GA)** to solve a constrained variant of the classic knapsack problem. The algorithm finds an optimal selection of snack items while satisfying multiple constraints simultaneously.

## Problem Description

The project solves a **multi-constrained knapsack problem** with the following constraints:

1. **Value Constraint**: The sum of selected item values must be ≥ minimum threshold
2. **Weight Constraint**: The sum of selected item weights must be ≤ maximum capacity  
3. **Item Count Constraint**: The number of selected items must be within a specified range
4. **Individual Weight Constraint**: Each selected item cannot exceed its maximum available weight

### Key Features

- **Flexible Item Selection**: Items can be selected partially (any weight between 0 and maximum available)
- **Multi-objective Optimization**: Balances value maximization with constraint satisfaction
- **Robust Genetic Operations**: Implements uniform crossover and mutation with configurable probabilities
- **Adaptive Fitness Function**: Uses weighted coefficients to handle constraint violations

## Dataset

The algorithm works with a snack dataset (`snacks.csv`) containing 19 different items:

| Sample Items | Available Weight | Value |
|--------------|------------------|-------|
| MazMaz | 10 | 10 |
| Jooj | 7 | 15 |
| Hot-Dog | 20 | 15 |
| Chocoroll | 9 | 12 |

## Algorithm Implementation

### Core Components

#### 1. **Chromosome Representation**
- Each chromosome contains genes for all 19 snack items
- Gene value represents the selected weight for each item (0 to max available weight)
- Enables partial item selection for fine-grained optimization

#### 2. **Fitness Function**
The fitness function balances value maximization with constraint penalties:

```
fitness = (total_value) / (|total_weight × W_items × W_value × W_weight²| + 1) × NORMALIZER
```

Where:
- `W_weight`: Weight constraint violation penalty (squared for emphasis)
- `W_value`: Value constraint violation penalty  
- `W_items`: Item count constraint violation penalty

#### 3. **Selection Method**
- **Rank-based selection** with roulette wheel technique
- Probability assignment: `rank_i = (i+1) / Σ(k)` for chromosome at position i
- Prevents premature convergence to suboptimal solutions

#### 4. **Genetic Operations**

**Uniform Crossover:**
- Creates two children by swapping genes between parents
- Probability-controlled gene exchange (default: 80%)

**Mutation:**
- Randomly modifies gene values with low probability (default: 1%)
- Essential for escaping local optima

### Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `POPULATION_SIZE` | 200 | Number of chromosomes per generation |
| `NUM_OF_ITERS` | 1000 | Maximum number of generations |
| `MUTATION_PROB` | 0.01 | Probability of gene mutation |
| `CROSSOVER_RATE` | 0.9 | Proportion of population undergoing crossover |
| `CROSSOVER_PROB` | 0.8 | Gene swap probability in uniform crossover |
| `MAX_WEIGHT` | 10 | Maximum total weight constraint |
| `MIN_VALUE` | 12 | Minimum total value constraint |
| `MIN_SNACKS_NUM` | 2 | Minimum number of items |
| `MAX_SNACKS_NUM` | 4 | Maximum number of items |

## File Structure

```
├── CA1.ipynb              # Main Jupyter notebook with implementation
├── snacks.csv             # Dataset with snack items, weights, and values
├── Description/           
│   └── AI-S03-A1.pdf     # Project requirements and specifications
└── README.md             # This documentation file
```

## Environment setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Prerequisites

```python
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
```

### Running the Algorithm

1. **Load the dataset:**
```python
df = pd.read_csv("snacks.csv")
```

2. **Create and run the genetic algorithm:**
```python
knapsack = Knapsack(POPULATION_SIZE, CROSSOVER_PROB, MUTATION_PROB, 
                   CROSSOVER_RATE, NUM_OF_FOODS, MAX_SNACKS_NUM, 
                   MIN_SNACKS_NUM, NUM_OF_ITERS, MAX_WEIGHT, MIN_VALUE)

average_fitnesses, final_population = knapsack.find_solution(True, True)
```

3. **Analyze results:**
```python
# Get the best solution
best_chromosome = final_population[-1][0]
selected_items = best_chromosome.get_genes()

# Display selected items and their weights
for i, weight in enumerate(selected_items):
    if weight > 0:
        print(f"{foods[i].get_name()}: {weight}")
```

### Example Output

```
Jooj: 7.0
Chocoroll: 2.8
Cookies: 0.2
Total Weight: 10.0
Total Value: 21.51
```

## Performance Analysis

### Convergence Visualization
The algorithm generates plots showing average fitness evolution across generations, demonstrating convergence behavior and optimization progress.

### Comparative Studies

The implementation includes experimental analysis comparing:

1. **With vs. Without Mutation**: Demonstrates mutation's role in escaping local optima
2. **With vs. Without Crossover**: Shows crossover's importance for population improvement
3. **Parameter Sensitivity**: Effects of different hyperparameter configurations

## Research Questions Addressed

The project provides detailed analysis of several key genetic algorithm concepts:

1. **Population Size Effects**: Impact of very large vs. very small initial populations
2. **Dynamic Population Sizing**: Effects on precision and convergence speed
3. **Genetic Operator Comparison**: Mutation vs. crossover effectiveness
4. **Optimization Strategies**: Methods for achieving faster convergence
5. **Convergence Issues**: Handling local optima and non-convergent scenarios
6. **Termination Criteria**: Detecting unsolvable problem instances

## Key Insights

- **Balanced Approach**: Both mutation and crossover are essential for optimal performance
- **Fitness Design**: Careful coefficient tuning in fitness function is crucial for constraint handling
- **Parameter Tuning**: Hyperparameter optimization significantly impacts solution quality
- **Constraint Handling**: Soft constraint violation through fitness penalties works better than hard constraints

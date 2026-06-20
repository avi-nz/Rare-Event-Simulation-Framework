# Rare Event Simulation Framework
Estimating extremely unlikely events using Monte Carlo simulation, importance sampling, adaptive sampling, and modern machine learning methods.

## Content & Documentation
This project is being documented publicly throughout development.

### Content:

#### 🎥 Full project breakdown video

#### 📱 Development shorts/reels

#### ✍️ Technical Substack articles

## Background Knowledge:
This project assumes familiarity with Monte Carlo simulation,
probability estimation, and statistical convergence.

The foundational experiments that motivated this project are
available in the Monte Carlo Playground repository: 

(insert Monte Carlo playground repo here)

## Overview
Rare events occur infrequently but often have outsized consequences.

Examples include:
* Financial crashes
* Power grid failures
* Aircraft failures
* Global pandemics
* Nuclear accidents

Standard Monte Carlo simulation becomes inefficient when estimating events with probabilities as small as:

$$
10^{-6}
$$

or lower.

This project explores increasingly sophisticated methods for efficiently and accurately estimating rare-event probabilities.

## Research Questions
This project investigates:

### Monte Carlo:
How many simulations are required before standard Monte Carlo becomes impractical?

This establishes a baseline for evaluating more advanced rare-event simulation techniques. As event probabilities decrease, standard Monte Carlo methods require an increasingly large number of samples to achieve accurate estimates. By measuring estimation error, variance, confidence intervals, and computational cost, we can quantify the limitations of brute-force simulation and compare the efficiency gains provided by alternative methods.

### Importance Sampling:
How much variance reduction can be achieved through importance sampling?

### Cross-Entropy Optimisation:
Can optimal sampling distributions be learned automatically?

### Adaptive Methods:
Can machine learning improve rare-event estimation?

### Quantitative Finance:
How accurately can rare-event methods estimate:
* Value at Risk
* Expected Shortfall
* Portfolio ruin probability

### Machine Learning:
Can rare-event techniques identify catastrophic model failures?

## Methodology

### Stage 1 - Standard Monte Carlo
Estimate probabilities directly.

**Example:**

$$
P(X>3)
$$

for

$$
X∼N(0,1)
$$

**Purpose:**

Establish baseline performance.

### Stage 2 - Variance Analysis
Measure:
* estimator variance
* confidence intervals
* simulation efficiency

**Question:**
- How quickly does Monte Carlo fail?

### Stage 3 - Importance Sampling
Modify sampling distributions to make rare events occur more frequently.

**Compare:**
```
Standard Monte Carlo
vs
Importance Sampling
```
**Metrics:**
* bias
* variance
* runtime

### Stage 4 - Splitting Methods
Implement:
* multilevel splitting
* subset simulation

**Question:**
- Can path-based methods outperform importance sampling?

### Stage 5 - Cross Entropy Method
Use optimisation to automatically learn efficient sampling distributions.

**Question:**
- Can we automatically discover optimal importance-sampling policies?

### Stage 6 - Adaptive Rare Event Simulation
Use machine learning to adapt sampling strategies during simulation.

**Methods:**
* reinforcement learning
* neural proposal distributions
* adaptive importance sampling

## Model Roadmap

### Model 0 - Naive Monte Carlo

🚧 **Planned** 🚧

**Features:**
- Direct simulation
- Frequency estimation

**Purpose:**
- Benchmark

### Model 1 — Variance Reduction

🚧 **Planned** 🚧

**Features:**
- Antithetic variables
- Control variates

**Question:**
- Can variance be reduced without changing distributions?

### Model 2 - Importance Sampling

🚧 **Planned** 🚧

**Features:**
- Distribution tilting
- Weighted estimators

**Question:**
- How much efficiency can be gained?

### Model 3 - Cross Entropy Optimisation

🚧 **Planned** 🚧

**Features:**
- Automatic proposal distribution search

**Question:**
- Can optimisation outperform manual design?

### Model 4 - Splitting Methods

🚧 **Planned** 🚧

**Features:**
- Multilevel splitting
- Subset simulation

**Question:**
- How effective are path-based approaches?

### Model 5 - Financial Risk Engine

🚧 **Planned** 🚧

**Features:**
- VaR estimation
- Expected shortfall
- Portfolio ruin probability

**Question:**
- Can rare-event methods improve tail-risk estimation?

### Model 6 - ML-Based Rare Event Discovery

🚧 **Planned** 🚧

**Features:**
- Adaptive sampling
- Neural importance distributions

**Question:**
- Can machine learning learn where rare events occur?

## Applications

### Reliability Engineering

**Estimate:**
```
P(System Failure)
```

**Examples:**
- Aircraft systems
- Power grids
- Manufacturing systems

### Quantitative Finance

**Estimate:**
```
P(Loss > Threshold)
```

**Examples:**
- Portfolio crashes
- Tail risk
- Liquidity events

### Machine Learning

**Estimate:**
```
P(Model Failure)
```

**Examples:**
- Adversarial examples
- Safety failures
- Distribution shifts

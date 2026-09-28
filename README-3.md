# Survival Analysis with Kaplan–Meier and DeepSurv

This project explores **classical and deep-learning approaches to survival analysis** using censored time-to-event data. It begins with Cox proportional hazards interpretation, implements the **Kaplan–Meier estimator and Greenwood confidence intervals from scratch**, and then builds a **DeepSurv-style neural network** trained with the Cox partial likelihood.

The analysis uses the breast cancer survival dataset provided by `scikit-survival` and evaluates how well a neural survival model can rank patients by risk.

## Project Overview

The project covers three major components:

1. **Cox proportional hazards interpretation**
   - Interpret Cox model coefficients and hazard ratios.
   - Compare risk between patient profiles.
   - Convert a reference survival probability into patient-specific survival and risk estimates.

2. **Kaplan–Meier survival estimation**
   - Implement the Kaplan–Meier estimator from scratch.
   - Account for right-censored observations.
   - Calculate 95% confidence intervals using Greenwood's formula.
   - Estimate survival probability at a clinically relevant time point.
   - Determine the median survival time.

3. **DeepSurv-style neural survival modeling**
   - Preprocess clinical and gene-expression features.
   - Build a feed-forward neural network that produces a scalar risk score.
   - Implement the negative Cox partial log-likelihood in PyTorch.
   - Train with early stopping.
   - Evaluate discrimination using Harrell's concordance index.
   - Estimate baseline cumulative hazard with the Breslow estimator.
   - Generate model-based survival curves.

## Dataset

The modeling portion uses:

```python
from sksurv.datasets import load_breast_cancer
```

The dataset contains **198 breast cancer patients** with clinical and gene-expression features.

| Characteristic | Value |
|---|---:|
| Patients | 198 |
| Raw features | 80 |
| Observed distant-metastasis events | 51 |
| Censored observations | 147 |
| Follow-up range | 4.1–299.2 months |
| Features after preprocessing | 82 |

The outcome is **time to distant metastasis**. Patients who did not experience the event during observed follow-up are treated as right-censored.

## Methods

### Cox Proportional Hazards

The Cox model is written as

```math
h(t \mid x) = h_0(t)\exp(\eta(x))
```

where $h_0(t)$ is the baseline hazard and $\eta(x)$ is the model's linear predictor.

The first section focuses on interpreting hazard ratios and converting model outputs into survival and risk estimates.

### Kaplan–Meier Estimator

The Kaplan–Meier survival estimate is implemented directly from event and censoring times:

```math
\hat{S}(t) = \prod_{t_j \leq t} \left(1-\frac{d_j}{n_j}\right)
```

where $d_j$ is the number of events at time $t_j$ and $n_j$ is the number of patients at risk immediately before that time.

A 95% confidence interval is computed using **Greenwood's standard error**.

### DeepSurv-Style Neural Network

After one-hot encoding categorical variables and standardizing features using the training set, the data are split into:

- **118 training patients**
- **40 validation patients**
- **40 test patients**

The neural network architecture is:

```text
Input (82 features)
        ↓
Linear(82, 64)
        ↓
      ReLU
        ↓
Linear(64, 32)
        ↓
      ReLU
        ↓
 Linear(32, 1)
        ↓
Scalar risk score
```

The network outputs a scalar risk score $\eta_\theta(x)$, where larger values correspond to higher predicted hazard.

Training uses the **negative Cox partial log-likelihood**:

```math
-\frac{1}{|D|} \sum_{i \in D} \left[\eta_i - \log\left(\sum_{j \in R_i} e^{\eta_j}\right)\right]
```

where $D$ is the set of observed events and $R_i$ is the risk set for patient $i$.

Optimization uses Adam with a learning rate of `1e-3`, weight decay, gradient clipping, and validation-based early stopping.

## Results

### Kaplan–Meier Analysis

At **60 months**, the full-cohort Kaplan–Meier estimate was:

```text
Survival probability: 0.821
95% CI: [0.767, 0.875]
```

The estimated median metastasis-free survival was approximately:

```text
236.1 months
```

### DeepSurv Test Performance

The final model achieved:

| Metric | Test Result |
|---|---:|
| Negative Cox partial log-likelihood | 3.2351 |
| Harrell C-index | 0.672 |

A C-index of **0.672** indicates that the model ranked the patient with the earlier observed event as higher risk in approximately 67.2% of comparable test-set pairs.

Because the test set contains only 40 patients, this result should be interpreted cautiously due to sampling variability.

### Model-Based Survival Curves

A Breslow baseline cumulative hazard estimator was used to convert neural-network risk scores into estimated survival curves.

At 60 months:

| Method | Survival Estimate |
|---|---:|
| Average DeepSurv prediction | 0.841 |
| Test-set Kaplan–Meier estimate | 0.825 |

The two estimates differed by approximately **1.6 percentage points** at 60 months and were generally similar through the portion of follow-up containing most observed events.

## Repository Structure

```text
.
├── DSB206_HW3_Part2_Survival_Analysis.ipynb
└── README.md
```

The Jupyter notebook contains the complete analysis, model implementation, visualizations, and interpretation.

## Installation

Create a Python environment and install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn scikit-survival torch jupyter
```

The project uses CUDA automatically when a compatible NVIDIA GPU is available; otherwise, it runs on CPU.

## Running the Project

Clone the repository and launch Jupyter:

```bash
git clone <repository-url>
cd <repository-directory>
jupyter notebook
```

Then open:

```text
DSB206_HW3_Part2_Survival_Analysis.ipynb
```

and run the cells in order.

## Key Skills Demonstrated

- Survival analysis with censored data
- Cox proportional hazards models
- Hazard-ratio interpretation
- Kaplan–Meier estimation
- Greenwood confidence intervals
- PyTorch neural-network development
- Custom survival-loss implementation
- DeepSurv-style modeling
- Harrell's concordance index
- Breslow baseline hazard estimation
- Model-based survival curve estimation
- Train/validation/test preprocessing without data leakage

## Limitations

The dataset is relatively small, particularly the 40-patient test set, so performance estimates may have substantial variance. The C-index measures **risk ranking rather than probability calibration**, meaning a model can rank patients well without necessarily producing perfectly calibrated survival probabilities.

A more extensive analysis could include repeated cross-validation, bootstrap confidence intervals for the C-index, comparison with a standard penalized Cox model, and formal survival calibration metrics.

## Technologies

**Python · NumPy · pandas · Matplotlib · scikit-learn · scikit-survival · PyTorch · Jupyter**

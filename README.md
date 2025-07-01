# Credit Risk Model for Bati Bank

This project implements an end-to-end machine learning solution to predict credit risk for a Buy-Now-Pay-Later service, using alternative data from an e-commerce platform.

## Credit Scoring Business Understanding

### The Influence of the Basel II Accord

The **Basel II Accord** forces banks to adopt more risk-sensitive approaches, meaning the capital they must hold is directly tied to the measured risk of their assets (like loans). This heavily influences our project by demanding a model that is not only accurate but also **highly interpretable and well-documented**. Regulators and internal auditors must be able to understand precisely *why* the model assigns a certain risk score. A "black box" model, regardless of its performance, presents a significant regulatory risk. Therefore, every step, from feature engineering to model choice, must be transparent and justifiable to prove our methodology is sound and compliant.

### The Necessity and Risks of a Proxy Variable

We lack a direct "default" label in our dataset. We cannot train a model without a target variable, so creating a **proxy variable** is essential. We are defining a proxy for high credit risk by identifying "disengaged" customers through **Recency, Frequency, and Monetary (RFM)** analysis. The hypothesis is that customers who transact less recently, less frequently, and with lower monetary value are less financially stable and thus represent a higher credit risk.

However, this carries a major business risk: **the proxy might be wrong**. A customer could be disengaged for reasons unrelated to creditworthiness (e.g., they moved, their needs changed). This can lead to two critical errors:
1.  **False Positives:** Rejecting creditworthy customers, resulting in lost revenue and customer dissatisfaction.
2.  **False Negatives:** Approving non-creditworthy customers, leading to loan defaults and financial losses.

### Model Trade-Offs: Simplicity vs. Complexity

In a regulated financial context, there is a fundamental trade-off between a simple, interpretable model like **Logistic Regression** and a complex, high-performance model like **Gradient Boosting**.

| **Factor** | **Logistic Regression (Simple)** | **Gradient Boosting (Complex)** |
| :--- | :--- | :--- |
| **Interpretability** | ✅ **High:** Easily explain how each feature impacts the final score. Preferred by regulators. | ❌ **Low:** Acts like a "black box." Difficult to explain its internal logic. |
| **Performance** | 🔶 **Moderate:** Might not capture complex, non-linear patterns in the data. | ✅ **High:** Excellent at finding intricate patterns, often leading to higher accuracy. |
| **Regulatory Risk** | **Low:** Its transparency makes it easier to validate and get approved. | **High:** Requires significant extra work (e.g., using SHAP) to justify its decisions to auditors. |

The key trade-off is accepting potentially lower (but still effective) predictive power in exchange for transparency and lower regulatory risk. For a new financial product, starting with a simple, interpretable model is often the most prudent strategy.
## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability
The Basel II Capital Accord emphasizes accurate measurement of credit risk to ensure that banks maintain sufficient capital buffers against potential losses. This regulatory focus requires credit scoring models to be **interpretable, transparent, and well-documented**, so that their predictions can be audited, explained to regulators, and justified in decision-making processes. Models must not only be predictive but also understandable to support responsible lending practices.

### 2. Need for a Proxy Variable
In this project, the dataset lacks a direct "default" label. To assess credit risk, we must create a **proxy target variable** that represents high-risk borrowers based on behavioral patterns (e.g., low transaction frequency, low monetary engagement). This is necessary to train predictive models.  
**Business risks** of using a proxy include:
- Misclassification of customer risk leading to incorrect lending decisions  
- Potential regulatory scrutiny if the proxy poorly reflects true default behavior  
- Bias in predictions if the proxy does not capture relevant financial behaviors

### 3. Model Trade-offs
In a regulated financial context, there is a trade-off between **simple, interpretable models** and **complex, high-performance models**:
- **Simple models** (e.g., Logistic Regression with Weight of Evidence encoding) are easier to explain, audit, and deploy. They provide transparency, which is crucial for regulatory compliance.
- **Complex models** (e.g., Gradient Boosting, Random Forests) may achieve higher predictive accuracy but are often opaque ("black-box"). They require additional techniques like SHAP values or LIME for interpretability.  
The choice of model depends on the balance between predictive performance and regulatory accountability.

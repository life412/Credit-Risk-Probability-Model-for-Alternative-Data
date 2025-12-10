Regulatory & Methodological Considerations

This project follows internationally recognized guidance on credit risk modeling, including concepts from:

Statistica Sinica research on credit scoring methodology

Hong Kong Monetary Authority (HKMA) Alternative Credit Scoring Guidelines

World Bank Credit Scoring Framework

Corporate Finance Institute credit risk learning modules

Risk-Officer industry guidance

These sources consistently emphasize risk measurement, documentation, interpretability, stability, and governance.
The following subsection answers three key regulatory questions relevant for this project.

1. How Basel II’s focus on risk measurement increases our need for interpretability and documentation

Basel II requires financial institutions to measure and justify their risk assessments, especially when using internal rating–based (IRB) approaches. The references emphasize that:

Regulators must understand how the model produces PD (probability of default).

Banks must be able to justify every transformation, predictor, cutoff, and assumption in the scoring system.

All processes — data cleaning, binning, WoE transformations, exclusions, model selection, and validation — must be fully traceable and reproducible.

Supervisors expect risk models to be transparent, because credit decisions affect capital requirements and consumer fairness.

Because of this, Basel II pushes organizations toward models that are:

Interpretable

Stable over time

Explainable to non-technical oversight bodies

Well-documented and supported by validation evidence

In short:
Basel II transforms credit scoring from a pure data science task into a regulated risk measurement process. This makes documentation, clarity, and interpretability non-negotiable.

2. Why we must create a proxy default variable, and the business risks associated with it

Many real-world financial datasets do not contain a clean, direct “default” label — especially early in a product’s life cycle or in alternative data environments. The references highlight that a clear definition of default is essential for:

Model training

Capital estimation

Portfolio segmentation

Validation and backtesting

Why a proxy is required

A credit risk model must learn the relationship between customer characteristics and a binary event.
If actual default data is missing, inconsistent, or delayed, we must construct a proxy default definition, such as:

90+ days past due (90 DPD)

Charge-off or write-off

Severely delinquent status

Collections escalation

These proxy definitions are commonly used by financial institutions and discussed in HKMA and World Bank guidance as practical alternatives when exact default records are not available.

Business risks of relying on a proxy

Using an imperfect proxy introduces several risks:

Label inaccuracy: Some customers labeled “default” by the proxy may cure later, and some genuinely risky customers may never hit the proxy threshold.

Calibration errors: PD estimates may be biased upward or downward, affecting approval rates, pricing, and risk models.

Portfolio mis-segmentation: Wrong risk segmentation can lead to poor credit decisions, higher losses, or rejection of good applicants.

Regulatory challenges: If the proxy is not clearly justified, documented, and stable across time, supervisors may reject the model.

Fairness concerns: A proxy influenced by operational behavior (e.g., payment channel differences) may unintentionally introduce bias.

Thus, creating a proxy is necessary — but must be handled with clear definitions, justification, and monitoring.

3. Trade-offs between a simple, interpretable model and a complex, high-performance model in regulated credit risk

The references consistently explain that regulated credit scoring requires a balance between:

Predictive accuracy

Interpretability

Supervisory acceptance

Operational stability

Simple, interpretable model: Logistic Regression + WoE

Advantages:

Highly interpretable — every variable’s effect is clear.

Well aligned with the classical scorecard design recommended by the World Bank and HKMA.

Easy to document, validate, and monitor.

Stable across time when built with monotonic WoE bins.

Preferred by regulators due to transparency.

Drawbacks:

Lower accuracy in complex data environments.

Limited ability to capture nonlinear patterns unless engineered manually.

Complex, high-performance model: Gradient Boosting (GBM, XGBoost, LightGBM)

Advantages:

Significantly higher predictive performance.

Automatically captures nonlinearities and feature interactions.

Useful for challenger models and portfolio analytics.

Drawbacks in a regulated environment:

Harder to explain to risk committees and regulators.

More challenging to validate and monitor.

Requires heavier governance, including stability tests, explainability analyses (e.g., SHAP), and periodic recalibration.

May be rejected for use in core credit decisions if insufficiently transparent.

The regulatory trade-off

In regulated credit environments (Basel II/III), the preferred production model is often:

Interpretable (Logistic Regression)

Supported by WoE binning

Fully documented and reproducible

Complex models may be used as benchmark or challenger models, but not always as the primary underwriting engine.

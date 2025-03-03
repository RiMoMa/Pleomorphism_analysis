import shap

def feature_importance(model, X):
    """Aplica SHAP para selección de características."""
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)


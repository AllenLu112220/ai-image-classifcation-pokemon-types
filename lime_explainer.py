import lime
import lime.lime_tabular

def explain_model_prediction(model, X_train, X_test, index=0):
    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['Second Wins', 'First Wins'], discretize_continuous=True)
    
    # Explain a single prediction
    exp = explainer.explain_instance(X_test.iloc[index].values, model.predict_proba, num_features=6)
    
    return exp

import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

# Step 1: Data Collection
def fetch_pokemon_data():
    # Fetch type effectiveness data (you may replace this with a custom dataset)
    effectiveness_data = {
        'type1': ['fire', 'water', 'grass', 'electric', 'psychic', 'fighting', 'rock', 'ground', 'ghost', 'bug'],
        'type2': ['grass', 'fire', 'water', 'ground', 'bug', 'flying', 'fighting', 'rock', 'fairy', 'dark'],
        'effectiveness': [2.0, 0.5, 1.0, 1.5, 1.0, 1.0, 0.5, 1.0, 1.0, 2.0]  # Example effectiveness values
    }
    return pd.DataFrame(effectiveness_data)

# Step 2: Load Data
df = fetch_pokemon_data()

# Step 3: Data Preprocessing
X = df[['type1', 'type2']]
y = df['effectiveness']

# One-hot encoding of the categorical variables
X_encoded = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Step 4: Model Development
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 5: Explainability Setup
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Step 6: Streamlit UI
st.title("Pokémon Type Effectiveness Predictor")

# User input for Pokémon types
type1 = st.selectbox("Select Type 1", df['type1'].unique())
type2 = st.selectbox("Select Type 2", df['type2'].unique())

if st.button("Predict Effectiveness"):
    input_data = pd.DataFrame([[type1, type2]], columns=['type1', 'type2'])
    input_encoded = pd.get_dummies(input_data).reindex(columns=X_encoded.columns, fill_value=0)
    prediction = model.predict(input_encoded)

    st.write(f"Predicted Effectiveness: {prediction[0]:.2f}")

    # Display SHAP values for explanation
    shap_values_input = explainer.shap_values(input_encoded)
    
    # Create a bar plot of SHAP values
    st.subheader("SHAP Values")
    plt.figure()
    shap.summary_plot(shap_values_input, input_encoded, show=False)
    st.pyplot(plt)

if st.button("Show Feature Importance"):
    feature_importance = model.feature_importances_
    feature_names = X_encoded.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    st.bar_chart(importance_df.set_index('Feature'))

# Run the Streamlit app using: streamlit run pokemon_predictor.py

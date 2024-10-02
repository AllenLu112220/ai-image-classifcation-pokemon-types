import streamlit as st
from preprocessing import load_data, prepare_battle_data, get_features_and_target
from train_model import train_model
from lime_explainer import explain_model_prediction

# Streamlit interface
st.title("Pokémon Battle Outcome Predictor with LIME")

# Load Pokémon and battle data
pokemon_file = 'pokemon_data.csv'
battle_file = 'battle_data.csv'
pokemon_df, battle_df = load_data(pokemon_file, battle_file)

# Merge and prepare data
battles_df = prepare_battle_data(pokemon_df, battle_df)

# Prepare features and target
X, y = get_features_and_target(battles_df)

# Train model
st.write("Training model...")
model, accuracy, X_train, X_test, y_train, y_test = train_model(X, y)
st.write(f"Model Accuracy: {accuracy:.2f}")

# LIME explanation for a specific prediction
st.write("Explaining model predictions with LIME...")
index = st.number_input('Select a test sample index for LIME explanation', min_value=0, max_value=len(X_test)-1, value=0)
exp = explain_model_prediction(model, X_train, X_test, index=index)

# Display LIME explanation
st.write("LIME Explanation:")
exp.show_in_notebook(show_table=True)

# Interactive inputs for Pokémon stats
st.write("Predict battle outcome between two custom Pokémon:")

pokemon1_stats = {
    'HP': st.number_input('First Pokémon HP', value=45),
    'Attack': st.number_input('First Pokémon Attack', value=49),
    'Defense': st.number_input('First Pokémon Defense', value=49),
    'Sp. Atk': st.number_input('First Pokémon Sp. Atk', value=65),
    'Sp. Def': st.number_input('First Pokémon Sp. Def', value=65),
    'Speed': st.number_input('First Pokémon Speed', value=45)
}

pokemon2_stats = {
    'HP': st.number_input('Second Pokémon HP', value=39),
    'Attack': st.number_input('Second Pokémon Attack', value=52),
    'Defense': st.number_input('Second Pokémon Defense', value=43),
    'Sp. Atk': st.number_input('Second Pokémon Sp. Atk', value=60),
    'Sp. Def': st.number_input('Second Pokémon Sp. Def', value=50),
    'Speed': st.number_input('Second Pokémon Speed', value=65)
}

# Calculate stat differences
input_data = [[
    pokemon1_stats['HP'] - pokemon2_stats['HP'],
    pokemon1_stats['Attack'] - pokemon2_stats['Attack'],
    pokemon1_stats['Defense'] - pokemon2_stats['Defense'],
    pokemon1_stats['Sp. Atk'] - pokemon2_stats['Sp. Atk'],
    pokemon1_stats['Sp. Def'] - pokemon2_stats['Sp. Def'],
    pokemon1_stats['Speed'] - pokemon2_stats['Speed']
]]

# Predict outcome
prediction = model.predict(input_data)
st.write(f"Predicted winner: {'First Pokémon' if prediction[0] == 1 else 'Second Pokémon'}")

import pandas as pd

def load_data(pokemon_file, battle_file):
    pokemon_df = pd.read_csv(pokemon_file)
    battle_df = pd.read_csv(battle_file)
    return pokemon_df, battle_df

def get_pokemon_stats(pokemon_id, pokemon_df):
    return pokemon_df[pokemon_df['id'] == pokemon_id].iloc[0]

def prepare_battle_data(pokemon_df, battle_df):
    battles = []
    
    for _, row in battle_df.iterrows():
        first_pokemon = get_pokemon_stats(row['First_pokemonID'], pokemon_df)
        second_pokemon = get_pokemon_stats(row['Second_pokemonID'], pokemon_df)
        
        battle = {
            'First_pokemonID': row['First_pokemonID'],
            'Second_pokemonID': row['Second_pokemonID'],
            'Winner': row['Winner']
        }
        
        for stat in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']:
            battle[f'First_{stat}'] = first_pokemon[stat]
            battle[f'Second_{stat}'] = second_pokemon[stat]
            battle[f'Diff_{stat}'] = first_pokemon[stat] - second_pokemon[stat]
        
        battles.append(battle)
    
    battles_df = pd.DataFrame(battles)
    return battles_df

def get_features_and_target(battles_df):
    X = battles_df[['Diff_HP', 'Diff_Attack', 'Diff_Defense', 'Diff_Sp. Atk', 'Diff_Sp. Def', 'Diff_Speed']]
    y = (battles_df['Winner'] == battles_df['First_pokemonID']).astype(int)
    return X, y

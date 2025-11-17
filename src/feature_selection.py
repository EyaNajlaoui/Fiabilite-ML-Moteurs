import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns

def select_important_features(train_df, test_df, target='RUL', n_features=15):
    feature_columns = [f'sensor{i}' for i in range(1, 22)] + ['op_regime']
    
    X_train = train_df[feature_columns]
    y_train = train_df[target]
    X_test = test_df[feature_columns]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    selector = SelectKBest(score_func=f_regression, k=n_features)
    selector.fit(X_train, y_train)
    
    selected_features = feature_importance.head(n_features)['feature'].tolist()
    
    print(f"Features selectionnes ({n_features} plus importants):")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(n_features)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {n_features} Features les plus importantes')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance.png', dpi=300)
    plt.show()
    
    return selected_features

def prepare_selected_sequences(data, selected_features, sequence_length=30, target='RUL'):
    sequences = []
    targets = []
    
    for unit in data['unit'].unique():
        unit_data = data[data['unit'] == unit].sort_values('cycle')
        
        if len(unit_data) <= sequence_length:
            continue
            
        if target not in unit_data.columns:
            continue
            
        features = unit_data[selected_features].values
        rul_values = unit_data[target].values
        
        for i in range(len(unit_data) - sequence_length):
            sequences.append(features[i:(i + sequence_length)])
            targets.append(rul_values[i + sequence_length])
    
    if len(sequences) == 0:
        print("Aucune sequence creee")
        return np.array([]), np.array([])
    
    return np.array(sequences), np.array(targets)
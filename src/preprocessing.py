import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def create_sequences_with_features(data, selected_features, sequence_length=30, target='RUL'):
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
    
    print(f"Sequences creees: {len(sequences)}")
    return np.array(sequences), np.array(targets)

def prepare_classification_data(train_df, test_with_rul, critical_threshold=30):
    train_df['critical'] = (train_df['RUL'] <= critical_threshold).astype(int)
    test_with_rul['critical'] = (test_with_rul['rul'] <= critical_threshold).astype(int)
    
    feature_columns = [f'sensor{i}' for i in range(1, 22)] + ['op_setting1', 'op_setting2', 'op_setting3', 'op_regime']
    
    X_train = train_df[feature_columns]
    y_train = train_df['critical']
    X_test = test_with_rul[feature_columns]
    y_test = test_with_rul['critical']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'models/classification_scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def scale_sequences(X_train_seq, X_test_seq):
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        return X_train_seq, X_test_seq
        
    scaler = StandardScaler()
    X_train_reshaped = X_train_seq.reshape(-1, X_train_seq.shape[-1])
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(X_train_seq.shape)

    X_test_reshaped = X_test_seq.reshape(-1, X_test_seq.shape[-1])
    X_test_scaled = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(X_test_seq.shape)
    
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled
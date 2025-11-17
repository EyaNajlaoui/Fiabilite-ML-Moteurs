import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import joblib

def create_lstm_model(sequence_length, n_features, learning_rate=0.05):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

def train_lstm_with_cv(X_train, y_train, sequence_length, n_features, n_splits=3, epochs=100):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_no = 1
    histories = []
    models = []
    
    for train_idx, val_idx in kf.split(X_train):
        print(f'\nEntrainement Fold {fold_no}')
        
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        model = create_lstm_model(sequence_length, n_features, learning_rate=0.05)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        
        history = model.fit(
            X_train_fold, y_train_fold,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        histories.append(history)
        models.append(model)
        fold_no += 1
    
    return models, histories

def evaluate_ensemble(models, X_test, y_test):
    predictions = []
    
    for model in models:
        y_pred = model.predict(X_test, verbose=0).flatten()
        predictions.append(y_pred)
    
    y_pred_ensemble = np.mean(predictions, axis=0)
    
    mse = mean_squared_error(y_test, y_pred_ensemble)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_ensemble)
    r2 = r2_score(y_test, y_pred_ensemble)
    
    print("=== PERFORMANCE DU MODELE LSTM (Ensemble) ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
    
    return y_pred_ensemble, metrics

def plot_cv_history(histories):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(['loss', 'mae', 'val_loss', 'val_mae']):
        for fold_idx, history in enumerate(histories):
            if metric in history.history:
                axes[i].plot(history.history[metric], label=f'Fold {fold_idx+1}')
        
        axes[i].set_title(f'Evolution {metric.upper()} - Cross Validation')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.upper())
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/lstm_cv_training.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(y_test, y_pred, metrics):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Vraies valeurs RUL')
    plt.ylabel('Predictions RUL')
    plt.title(f'Predictions vs Vraies valeurs\nR2 = {metrics["r2"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Residus')
    plt.title('Analyse des Residus')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/lstm_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_models(models, file_prefix='lstm_model_fold'):
    for i, model in enumerate(models):
        model.save(f'models/{file_prefix}_{i+1}.h5')
    
    print(f"{len(models)} modeles sauvegardes")

def load_models(file_prefix='lstm_model_fold', n_models=3):
    models = []
    for i in range(n_models):
        model = tf.keras.models.load_model(f'models/{file_prefix}_{i+1}.h5')
        models.append(model)
    
    return models
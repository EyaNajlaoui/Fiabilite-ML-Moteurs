import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_rul_evolution(train_df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, unit_id in enumerate([1, 10, 20, 30]):
        unit_data = train_df[train_df['unit'] == unit_id]
        axes[i].plot(unit_data['cycle'], unit_data['RUL'], linewidth=2)
        axes[i].set_title(f'Evolution RUL - Moteur {unit_id}')
        axes[i].set_xlabel('Cycle')
        axes[i].set_ylabel('RUL')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/rul_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sensor_trends(train_df, sensor_columns, n_sensors=6):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(min(n_sensors, len(sensor_columns))):
        sensor = sensor_columns[i]
        sample_units = train_df['unit'].unique()[:5]
        
        for unit in sample_units:
            unit_data = train_df[train_df['unit'] == unit]
            axes[i].plot(unit_data['cycle'], unit_data[sensor], alpha=0.7, label=f'Unit {unit}')
        
        axes[i].set_title(f'Tendance - {sensor}')
        axes[i].set_xlabel('Cycle')
        axes[i].set_ylabel('Valeur')
        axes[i].grid(True, alpha=0.3)
        
        if i == 0:
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('results/figures/sensor_trends.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_regime_performance(test_df_seq, y_test_seq, y_pred):
    test_df_seq = test_df_seq.copy()
    test_df_seq['predicted_rul'] = y_pred
    test_df_seq['true_rul'] = y_test_seq
    
    regimes = sorted(test_df_seq['op_regime'].unique())
    n_regimes = len(regimes)
    
    fig, axes = plt.subplots(1, n_regimes, figsize=(5*n_regimes, 5))
    
    if n_regimes == 1:
        axes = [axes]
    
    for i, regime in enumerate(regimes):
        regime_data = test_df_seq[test_df_seq['op_regime'] == regime]
        
        axes[i].scatter(regime_data['true_rul'], regime_data['predicted_rul'], alpha=0.6)
        axes[i].plot([regime_data['true_rul'].min(), regime_data['true_rul'].max()], 
                    [regime_data['true_rul'].min(), regime_data['true_rul'].max()], 'r--', lw=2)
        axes[i].set_xlabel('Vraies valeurs RUL')
        axes[i].set_ylabel('Predictions RUL')
        axes[i].set_title(f'Regime {regime} - Performance RUL')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/regime_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
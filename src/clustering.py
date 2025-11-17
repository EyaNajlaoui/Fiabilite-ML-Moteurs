import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

def find_optimal_clusters(op_settings_scaled, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(op_settings_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Methode du Coude pour KMeans')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('WCSS')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return wcss

def apply_kmeans(train_df, test_df, n_clusters=3):
    op_settings = ['op_setting1', 'op_setting2', 'op_setting3']
    
    scaler_op = StandardScaler()
    op_settings_scaled_train = scaler_op.fit_transform(train_df[op_settings])
    op_settings_scaled_test = scaler_op.transform(test_df[op_settings])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_df['op_regime'] = kmeans.fit_predict(op_settings_scaled_train)
    test_df['op_regime'] = kmeans.predict(op_settings_scaled_test)
    
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(scaler_op, 'models/op_scaler.pkl')
    
    print("Repartition des regimes operationnels :")
    print(train_df['op_regime'].value_counts().sort_index())
    
    return train_df, test_df

def plot_clusters_3d(train_df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(train_df['op_setting1'], train_df['op_setting2'], 
                        train_df['op_setting3'], c=train_df['op_regime'], 
                        cmap='viridis', alpha=0.6, s=10)
    
    ax.set_xlabel('Op Setting 1')
    ax.set_ylabel('Op Setting 2')
    ax.set_zlabel('Op Setting 3')
    ax.set_title('Clusters des Regimes Operationnels')
    
    plt.colorbar(scatter, ax=ax)
    plt.savefig('results/figures/clustering_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
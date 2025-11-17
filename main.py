import os
import numpy as np
import pandas as pd
from src.data_loader import load_data, add_rul_to_train, add_rul_to_test
from src.preprocessing import create_sequences_with_features, prepare_classification_data, scale_sequences
from src.clustering import apply_kmeans
from src.classification import train_classifiers, plot_classification_results, plot_confusion_matrices, plot_roc_curves
from src.feature_selection import select_important_features, prepare_selected_sequences
from src.rul_prediction import train_lstm_with_cv, evaluate_ensemble, plot_cv_history, plot_predictions, save_models
from src.visualization import (
    plot_rul_evolution_4_engines, plot_sensor_correlation_heatmap, plot_6_operational_regimes,
    plot_weibull_mle_vs_mom, plot_kaplan_meier_median, plot_weibull_vs_kaplan,
    plot_reliability_failure_rate, create_fmds_table
)
from src.utils import save_metrics
import joblib

def main():
    print("Initialisation du projet de prediction RUL...")
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    print("\n1. Chargement des donnees...")
    train_df, test_df, rul_df = load_data()
    
    print("\n2. Ajout des RUL...")
    train_df = add_rul_to_train(train_df)
    test_df, test_with_rul = add_rul_to_test(test_df, rul_df)
    
    print(f"Train avec RUL: {train_df.shape}")
    print(f"Test avec RUL: {test_with_rul.shape}")
    
    print("\n3. Visualisations principales...")
    plot_rul_evolution_4_engines(train_df)
    plot_sensor_correlation_heatmap(train_df)
    
    print("\n4. Analyse Weibull et fiabilite...")
    shape_mle, loc_mle, scale_mle = plot_weibull_mle_vs_mom(train_df)
    kmf = plot_kaplan_meier_median(train_df)
    plot_weibull_vs_kaplan(shape_mle, loc_mle, scale_mle, kmf)
    plot_reliability_failure_rate(shape_mle, loc_mle, scale_mle)
    
    print("\n5. Tableau FMDS...")
    fmds_table = create_fmds_table(shape_mle, loc_mle, scale_mle)
    
    print("\n6. Clustering des regimes operationnels...")
    train_df = plot_6_operational_regimes(train_df)
    train_df, test_df = apply_kmeans(train_df, test_df, n_clusters=3)
    
    test_with_rul = test_with_rul.merge(
        test_df[['unit', 'cycle', 'op_regime']], 
        on=['unit', 'cycle'], 
        how='left'
    )
    
    print("\n7. Selection des features importantes...")
    selected_features = select_important_features(train_df, test_df, n_features=15)
    
    print("\n8. Classification - Etat sain vs critique...")
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = prepare_classification_data(train_df, test_with_rul)
    
    print(f"Donnees classification - X_train: {X_train_clf.shape}, X_test: {X_test_clf.shape}")
    
    classification_results = train_classifiers(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    
    plot_classification_results(classification_results, y_test_clf)
    plot_confusion_matrices(classification_results, y_test_clf)
    plot_roc_curves(classification_results, y_test_clf)
    
    classification_metrics = {}
    for name, result in classification_results.items():
        classification_metrics[name] = {
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'roc_auc': result['roc_auc']
        }
    
    save_metrics(classification_metrics, 'results/metrics/classification_metrics.json')
    
    print("\n9. Prediction RUL avec LSTM (3-Fold CV)...")
    sequence_length = 30
    
    print("Creation des sequences d'entrainement avec features selectionnees...")
    X_train_seq, y_train_seq = prepare_selected_sequences(train_df, selected_features, sequence_length, 'RUL')
    
    print("Creation des sequences de test avec features selectionnees...")
    test_df_for_rul = test_df.copy()
    test_df_for_rul['RUL'] = test_df_for_rul.groupby('unit')['cycle'].transform('max') - test_df_for_rul['cycle']
    X_test_seq, y_test_seq = prepare_selected_sequences(test_df_for_rul, selected_features, sequence_length, 'RUL')
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        print("Aucune sequence creee. Passage a l'etape suivante.")
    else:
        print(f"Sequences d'entrainement: {X_train_seq.shape}")
        print(f"Sequences de test: {X_test_seq.shape}")
        
        X_train_scaled, X_test_scaled = scale_sequences(X_train_seq, X_test_seq)
        
        n_features = X_train_scaled.shape[2]
        
        print(f"\nEntrainement LSTM avec 3-Fold Cross Validation")
        print(f"Features: {n_features}, Learning Rate: 0.05, Early Stopping: 15 patience")
        
        models, histories = train_lstm_with_cv(
            X_train_scaled, y_train_seq, 
            sequence_length, n_features, 
            n_splits=3, epochs=100
        )
        
        plot_cv_history(histories)
        save_models(models)
        
        print("\nEvaluation de l'ensemble des modeles...")
        y_pred, rul_metrics = evaluate_ensemble(models, X_test_scaled, y_test_seq)
        save_metrics(rul_metrics, 'results/metrics/rul_metrics.json')
        
        plot_predictions(y_test_seq, y_pred, rul_metrics)
    
    print("\n=== SYNTHESE DES RESULTATS ===")
    best_classifier = max(classification_metrics, key=lambda x: classification_metrics[x]['f1'])
    print(f"Meilleur classificateur: {best_classifier}")
    print(f"  - Accuracy: {classification_metrics[best_classifier]['accuracy']:.4f}")
    print(f"  - F1-Score: {classification_metrics[best_classifier]['f1']:.4f}")
    print(f"  - ROC-AUC: {classification_metrics[best_classifier]['roc_auc']:.4f}")
    
    if 'rul_metrics' in locals():
        print(f"\nPerformance RUL LSTM:")
        print(f"  - MAE: {rul_metrics['mae']:.2f} cycles")
        print(f"  - RMSE: {rul_metrics['rmse']:.2f} cycles")
        print(f"  - R2: {rul_metrics['r2']:.4f}")
    
    print(f"\nFeatures utilisees dans LSTM: {len(selected_features)}")
    print("Top 5 features:", selected_features[:5])
    
    print("\nProjet termine avec succes!")

if __name__ == "__main__":
    main()
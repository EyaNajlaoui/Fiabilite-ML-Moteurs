import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def train_classifiers(X_train, X_test, y_train, y_test):
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"Entrainement {name}...")
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': clf,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        joblib.dump(clf, f'models/classifier_{name.lower()}.pkl')
        
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    return results

def plot_classification_results(results, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [results[name][metric] for name in results.keys()]
        axes[i].bar(results.keys(), values, color=['blue', 'green', 'red'])
        axes[i].set_title(f'{metric_name} par Modele')
        axes[i].set_ylabel(metric_name)
        axes[i].set_ylim(0, 1)
        
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/figures/classification_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'Matrice de Confusion - {name}')
        axes[idx].set_xlabel('Prediction')
        axes[idx].set_ylabel('Reel')
    
    plt.tight_layout()
    plt.savefig('results/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(results, y_test):
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbes ROC - Comparaison des Modeles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
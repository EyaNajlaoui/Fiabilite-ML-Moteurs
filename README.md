# Prédiction de la RUL & Classification de Santé des Moteurs d’Avion  
## Projet basé sur les données NASA C-MAPSS (FD002)

---

## Présentation du Projet

Ce projet implémente un système avancé de maintenance prédictive des moteurs d’avion utilisant le dataset NASA C-MAPSS.  
Il combine l’ingénierie de fiabilité, l’analyse de survie, l’apprentissage automatique et le Deep Learning pour prédire la Durée de Vie Résiduelle (RUL) et classifier l’état de santé des moteurs.

---

## Fonctionnalités Principales

- **Prédiction RUL** : estimation des cycles restants avant défaillance  
- **Classification de santé** : Sain vs Critique  
- **Analyse de fiabilité** : Modèles paramétriques (Weibull)  
- **Analyse de survie** : Kaplan-Meier  
- **Clustering** : Détection des régimes opérationnels  
- **Validation croisée** : 3-fold  
- **Sélection de variables** : Capteurs les plus prédictifs  

---

## Description du Dataset — NASA C-MAPSS (FD002)

- **Unités d'entraînement** : 260 moteurs  
- **Unités de test** : 259 moteurs  
- **Capteurs** : 21 capteurs moteur  
- **Paramètres opérationnels** : 3 réglages  
- **Conditions opérationnelles** : 6 conditions différentes  
- **Données** : Séries temporelles multivariées  

---

## Méthodologie Implémentée

---

## Approche Paramétrique — Ingénierie de Fiabilité

### Analyse de la Distribution de Weibull
- Ajustement MLE  
- Estimation MOM  
- Calculs :  
  - MTBF  
  - Taux de défaillance λ(t)  
  - Fiabilité R(t)  
  - Quantiles : T10, T50, T90  

### Indicateurs FMDS
- Fiabilité à 100, 150 et 200 cycles  
- Disponibilité opérationnelle  
- Analyse des modes de défaillance  

---

## Approche Non-Paramétrique — Analyse de Survie

### Estimateur de Kaplan-Meier
- Fonction de survie empirique  
- Durée de vie médiane  
- Intervalles de confiance  
- Comparaison avec Weibull  

### Analyse complémentaire
- Courbes de survie  
- Fonctions cumulatives  
- Tests de comparaison  

---

## Apprentissage Automatique — Classification de Santé

### Modèles utilisés
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)

### **Meilleur modèle : Random Forest**
- **Accuracy** : 96.63%  
- **Precision** : 97.65%  
- **Recall** : 96.63%  
- **F1-Score** : 97.01%  
- **AUC ROC** : 98.06%  

---

## Deep Learning — Prédiction RUL (LSTM)

### Architecture du modèle
- 3 couches LSTM : **128 → 64 → 32 neurones**  
- Fenêtre temporelle : 30 cycles  
- Optimiseur : Adam (LR = 0.05)  
- Régularisation : Dropout + BatchNorm  
- Early stopping : patience = 15  
- Sélection des 15 variables les plus importantes  

### Performances
- **R² : 0.8317**  
- **RMSE : 24.55**  
- **MAE : 19.03**  

---

## Clustering des Régimes Opérationnels

- Méthode : K-Means  
- Nombre de clusters : 6  
- Standardisation  
- Visualisation 3D  

---

## Architecture Technique

### Prétraitement
- Nettoyage des données  
- Calcul RUL  
- Séquences temporelles (fenêtre=30)  
- Normalisation (StandardScaler)  
- Gestion des valeurs manquantes  

### Ingénierie des Variables
- Sélection des capteurs  
- Intégration des clusters  
- Extraction de features statistiques  
- Analyse de corrélation  

### Métriques & Validation
- **Classification** : Accuracy, Precision, Recall, F1, ROC-AUC  
- **Régression** : MSE, RMSE, MAE, R²  
- Cross-validation (3-fold)  
- Early stopping  

---

## Installation

```bash
git clone https://github.com/EyaNajlaoui/Fiabilite-ML-Moteurs.git
cd Fiabilite-ML-Moteurs
pip install -r requirements.txt

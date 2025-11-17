# Prédiction RUL et Classification de Santé des Moteurs d'Avion

## Présentation du Projet

Ce projet implémente un système de surveillance de santé des moteurs d'avion utilisant les données NASA C-MAPSS. Le système combine des approches paramétriques et non-paramétriques pour la prédiction de la Durée de Vie Résiduelle (RUL) et la classification de l'état de santé des moteurs, démontrant l'application de techniques statistiques et de méthodes d'apprentissage automatique pour la maintenance prédictive dans l'industrie aéronautique.

## Fonctionnalités Principales

- **Prédiction RUL** : Estimation des cycles opérationnels restants avant défaillance
- **Classification de Santé** : Classification binaire de l'état des moteurs (Sain vs Critique)
- **Analyse de Fiabilité** : Ingénierie de fiabilité utilisant des modèles de survie
- **Clustering des Régimes** : Identification des modes opérationnels de vol
- **Validation Croisée** : Validation par 3 folds pour l'évaluation des modèles
- **Sélection de Variables** : Sélection des capteurs et paramètres les plus prédictifs

## Description des Données

### Jeu de Données NASA C-MAPSS FD002
- **Unités d'Entraînement** : 260 moteurs avec données jusqu'à défaillance
- **Unités de Test** : 259 moteurs avec séquences partielles
- **Capteurs** : 21 capteurs de surveillance mesurant des paramètres moteur
- **Paramètres Opérationnels** : 3 réglages représentant l'altitude, l'angle de manette et le Mach
- **Conditions Opérationnelles** : 6 conditions opérationnelles et modes de défaillance
- **Structure des Données** : Données de séries temporelles multivariées

## Méthodologie Implémentée

### Approche Paramétrique (Ingénierie de Fiabilité)

#### Analyse de Distribution de Weibull
- Ajustement par Maximum de Vraisemblance (MLE)
- Estimation des paramètres par Méthode des Moments (MOM)
- Calculs des indicateurs de fiabilité :
  - MTBF (Temps Moyen Entre Défaillances)
  - Analyse du taux de défaillance λ(t)
  - Courbes de fiabilité R(t)
  - Quantiles de durée de vie (T10, T50, T90)

#### Indicateurs de Performance FMDS
- Fiabilité à 100, 150, 200 cycles
- Analyse du taux de défaillance instantané
- Calculs de disponibilité opérationnelle
- Analyse des modes de défaillance

### Approche Non-Paramétrique (Analyse de Survie)

#### Estimateur de Kaplan-Meier
- Estimation de la fonction de survie empirique
- Calcul de la durée de vie médiane
- Intervalles de confiance
- Comparaison de modèles avec Weibull

#### Analyse de Survie
- Courbes de survie empiriques
- Fonctions de répartition cumulatives
- Tests de comparaison de modèles

### Implémentation d'Apprentissage Automatique

#### Classification de Santé (Sain vs Critique)
- **Modèles utilsés** :
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- **Meilleur Modèle : Random Forest Classifier**
## Performance du Random Forest :
- Accuracy : 96.63%
- Precision : 97.65%
- Recall : 96.63%
- F1-Score : 97.01%
- AUC ROC : 98.06%


#### Prédiction RUL (LSTM)
- **Architecture** : LSTM à 3 couches (128-64-32 neurones)
- **Validation Croisée** : 3 folds stratifiés
- **Optimisation** : Optimiseur Adam (Taux d'Apprentissage=0.05)
- **Régularisation** : Dropout, Normalisation par lots
- **Arrêt Précoce** : Patience de 15 époques
- **Sélection de Variables** : Top 15 des variables les plus importantes

#### Clustering des Régimes Opérationnels
- **K-Means** : 6 régimes opérationnels identifiés
- **Standardisation** : Normalisation des paramètres
- **Visualisation 3D** : Espace des paramètres opérationnels

## Architecture Technique

### Prétraitement des Données
- Nettoyage et validation des données capteurs
- Calcul du RUL à partir des données de cycles
- Création de séquences temporelles (fenêtre=30 cycles)
- Normalisation StandardScaler
- Gestion des valeurs manquantes

### Ingénierie des Variables
- Sélection des capteurs les plus informatifs
- Intégration des régimes opérationnels
- Création de caractéristiques statistiques (moyenne, écart-type, percentiles)
- Analyse de corrélation entre capteurs

### Validation et Métriques
- **Classification** : Exactitude, Précision, Rappel, F1-Score, ROC-AUC
- **Régression** : MSE, RMSE, MAE, R²
- **Validation Croisée** : 3 folds pour prévenir le surapprentissage
- **Arrêt Précoce** : Patience de 15 époques

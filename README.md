# Projet Machine Learning : Transcription en Direct de la Langue des Signes Française (LSF) 

## Description 

Ce projet vise à développer une architecture capable de transcrire en temps réel les lettres signées de la Langue des Signes Française (LSF) capturées par une caméra. L'objectif secondaire est de transcrire des phrases complètes à partir des lettres détectées. Ce projet a été conçu pour faciliter la communication entre les personnes malentendantes et celles ne connaissant pas la LSF.

## Fonctionnalités principales 
 
- Détection des articulations de la main avec **MediaPipe** .

- Extraction des caractéristiques géométriques des gestes (angles, orientation de la main, relation entre doigts).

- Classification des lettres signées à l'aide de modèles de Machine Learning (KNN, Random Forest, Logistic Regression).

- Combinaison des prédictions pour améliorer la précision.

- Algorithme de segmentation des mots pour générer des phrases cohérentes.
 
- Intégration expérimentale de **YOLOv7**  pour une détection améliorée des mains. (Optionnel, désactivé par défault)

## Résultats obtenus 
 
- Précision moyenne des modèles testés : 93 % sur les données de test.

- Segmentation des mots : Le meilleur algorithme a identifié correctement environ 69 % des mots.
 
- Détection des mains avec YOLOv7 :
  - Précision : 80 %

  - mAP@0.5 : 70 %

## Structure du projet 
  
- **Notebook d'entrainement des modèles de detection des lettres**  : letterdetect.ipynb

- **Notebook de test de la segmentation et la reconstruction syntaxique des phrases**  : reformerphrase.ipynb

- **Notebook d'entrainement de YOLOv7**  : yolo_src/train.ipynb

- **Code final** : live.py
  
## Prérequis 

- Python 3.9+
 
- Bibliothèques requises (listées dans `requirements.txt`):
  - Mediapipe

  - Scikit-learn

  - Numpy, Pandas

  - Pytorch (pour YOLOv7)

## Installation 
 
1. Clonez ce dépôt :

```bash
git clone https://github.com/arnaudcournil/LSF-Alphabet-Subtitles.git
cd LSF-Alphabet-Subtitles
```
 
2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation 
 
1. Lancez la transcription en temps réel :

```bash
python live.py
```
 
2. Activez l'option YOLOv7 en modifiant le booléen `USE_YOLO` dans `live.py`.

## Limites 

- Charge computationnelle élevée avec YOLOv7, nécessitant des GPU performants.

- Base de données limitée (339 images), ce qui peut restreindre la généralisation du modèle.

## Perspectives d'amélioration 

- Création d'un dataset plus étendu et diversifié.

- Développement d'un modèle basé sur la reconnaissance de mots complets plutôt que de lettres.

- Optimisation pour les dispositifs à faible puissance.

## Auteurs 

- Laurent Vong

- Arnaud Cournil

- Maxime Chappuis

## Sources et Références 

- Dataset utilisé : COCO-Hand, TV-Hand
 
- Documentation MediaPipe : [Guide de détection des points de repère de la main](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=fr)

- Vidéos d'entraînement : Voir la section "Sources" du rapport.


---
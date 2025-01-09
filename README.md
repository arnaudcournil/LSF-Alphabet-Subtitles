# Projet Machine Learning : Transcription en Direct de la Langue des Signes Française (LSF)

## Introduction
Ce projet vise à développer un système permettant de transcrire en temps réel les lettres de la Langue des Signes Française (LSF) captées via une webcam. L'objectif final est de transformer les séquences de lettres en phrases cohérentes.  
Étant donné l'absence de datasets disponibles pour la LSF, nous avons créé notre propre base de données à partir de photos et de captures d'écrans.

---

## Structure du Projet

### 1. Base de Données
- **Création du Dataset** :  
  - Photographies des membres de l'équipe réalisant chaque signe de l'alphabet.
  - Captures d'écran de vidéos YouTube montrant des lettres en LSF.  
- **Nettoyage des données** :  
  - Suppression des images de mauvaise qualité pour réduire les biais lors de l'entraînement.  
- **Résultat final** : 339 images étiquetées représentant les lettres de l'alphabet en LSF.

---

### 2. Approches et Modèles Utilisés
#### 2.1. MediaPipe
- Détection des articulations principales de la main (21 points clés) à partir des images.  
- Extraction des coordonnées des points d'intérêt pour les étapes suivantes.

#### 2.2. Prétraitement des Données
- Étapes spécifiques (détails à compléter dans le code source).  

#### 2.3. Modèle Simple
- Entraînement d'un modèle de classification simple pour la prédiction des lettres.  

#### 2.4. Combinaison de Modèles
- Fusion de plusieurs modèles pour améliorer la précision des prédictions.

#### 2.5. Algorithme de Séparation des Lettres
- Détection des limites entre les lettres pour produire des mots ou phrases lisibles.

---

### 3. Résultats
#### 3.1. Modèle Simple
- **Performances** : Résultats à compléter (accuracy, précision, rappel, etc.).  

#### 3.2. Modèles Combinés
- Comparaison des résultats par rapport au modèle simple.  

#### 3.3. Algorithme de Séparation des Lettres
- Tests de robustesse avec des phrases contenant ou non des espaces.

---

### 4. Nouvelle Approche : YOLOv7 + MediaPipe
Pour améliorer les performances :
- **YOLOv7** : Détection rapide et précise des mains dans une image.  
- **MediaPipe** : Segmentation fine des articulations de la main.  

#### Processus :
1. **Entraînement YOLOv7** : Sur le dataset COCO-Hand (25 000 images annotées).  
2. **Détection en temps réel** :  
   - YOLOv7 : Détection des boîtes englobantes des mains.  
   - MediaPipe : Segmentation des points clés des mains.  
3. **Prédiction des lettres** : Utilisation des modèles déjà entraînés.  

#### Points forts :
- Combinaison de la détection rapide (YOLOv7) et de la segmentation fine (MediaPipe).  
#### Limites :
- Complexité computationnelle élevée, nécessitant des ressources GPU importantes.

---

## Conclusion
Ce projet démontre la faisabilité d'un système de transcription en temps réel de la LSF à partir de données limitées. Une optimisation supplémentaire est nécessaire pour améliorer les performances en conditions réelles.

---

## Ressources et Références
### Vidéos Utilisées pour le Dataset :
- [Vidéo 1](https://youtu.be/jg5zXcN2tlY?si=d5CBL6WYuoOm_5nh)  
- [Vidéo 2](https://youtu.be/QrOdNX32HyA?si=vo3HW_GFmHjQBC-n)  
- [Vidéo 3](https://youtu.be/HGGmrxeZEGQ?si=xQCuBRZUaf3ib4DB)  
- [Vidéo 4](https://youtu.be/H_DZk-fDDV8?si=UiuCONzfeks4ZMsk)  
- [Vidéo 5](https://youtu.be/KfExLSjNGc4?si=KUXUovJ9i6MexMor)  
- [Vidéo 6](https://youtu.be/sK3NyDGAO48?si=4l587c_Gx-DCd2V5)  
- [Vidéo 7](https://youtu.be/XQEFR5YmIP4?si=ey5qbTOKjDEOyTPI)  
- [Vidéo 8](https://youtu.be/4QPmDdTcX6I?si=7NqMxjR7P2LIXFIa)  

---

## Auteurs
- Laurent Vong  
- Arnaud Cournil  
- Maxime Chappuis  

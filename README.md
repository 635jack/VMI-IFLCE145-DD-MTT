# 🔬 MTT-Distillation: Dataset Distillation by Matching Training Trajectories

[![Paper](https://img.shields.io/badge/Paper-CVPR%202022-blue)](https://arxiv.org/abs/2203.11932)
[![Presentation](https://img.shields.io/badge/Slides-Project%20Presentation-orange)](https://docs.google.com/presentation/d/1eSJU80N8AxmIcU6pbwFljB2u_6-8VQHDuSII4-ztjWM/edit?usp=sharing)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Buffers%20%26%20Logs-yellow)](https://huggingface.co/jack635/mtt-distillation-buffers)

Cette version épurée du projet **MTT (Matching Training Trajectories)** permet de reproduire et d'analyser la condensation de datasets (ex: CIFAR-10) en un nombre extrêmement réduit d'images synthétiques, tout en conservant une excellente performance de test.

## 🌟 Points Forts
*   **Performance** : Atteint **46.3%** d'accuracy sur CIFAR-10 avec seulement **1 image par classe** (IPC=1).
*   **Structure Intuitive** : Workflow divisé en 3 notebooks spécialisés pour une prise en main rapide.
*   **Ressources Externes** : Accès direct aux trajectoires expertes (buffers) pré-entraînées sur Hugging Face.

---

## 📂 Structure du Projet

Le dépôt est organisé autour de trois piliers principaux :

1.  **`distillation.ipynb`** : **Le Cœur du Projet.** Configurez vos hyperparamètres, chargez vos trajectoires expertes et lancez l'optimisation pour générer vos propres images distillées.
2.  **`visualization.ipynb`** : **Analyse Visuelle.** Explorez l'évolution des images synthétiques, les courbes d'apprentissage, et visualisez la correspondance des trajectoires via des projections PCA 2D/3D.
3.  **`benchmarking.ipynb`** : **Validation.** Évaluez la robustesse de vos données distillées sur différentes architectures (ConvNet, ResNet, VGG) et comparez les résultats avec les baselines du papier original.

---

## 🚀 Démarrage Rapide

### 1. Installation
```bash
git clone https://github.com/[ton-username]/mtt-distillation-clean.git
cd mtt-distillation-clean
pip install -r requirements.txt
```

### 2. Récupération des données
Les trajectoires expertes (obligatoires pour la distillation) ainsi que les logs de nos runs précédents sont disponibles ici :
👉 [**Hugging Face Repository**](https://huggingface.co/jack635/mtt-distillation-buffers)


---

## 📚 Références & Crédits
*   **Article Original** : [Cazenavette et al., CVPR 2022](https://arxiv.org/abs/2203.11932)
*   **Présentation du Projet** : [Consulter les slides](https://docs.google.com/presentation/d/1eSJU80N8AxmIcU6pbwFljB2u_6-8VQHDuSII4-ztjWM/edit?usp=sharing)

*Ce projet a été réalisé dans le cadre du module **VMI (Modélisation de Systèmes Intelligents)**.*

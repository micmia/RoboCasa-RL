## Roadmap et Plan d’Implémentation

L’objectif est de piloter le robot **PandaOmron** afin de réaliser une **tâche atomique** dans RoboCasa, en excluant explicitement les problématiques de navigation.

---

### Phase 1 : Cadrage et Mise en Place

- **Choix de la tâche**  
Sélectionner une tâche atomique parmi les suivantes : ouverture/fermeture de porte ou de couvercle, insertion, *pick & place*, appui sur bouton, glissement de tiroir/rack, rotation de levier ou de bouton.  
👉 Privilégier une tâche simple pour une première implémentation.
- **Validation de l’environnement**  
Vérifier que :
  - l’environnement RoboCasa se lance correctement avec le robot PandaOmron ;
  - une politique aléatoire (*random policy*) peut être exécutée sans erreur ;
  - les observations et actions sont correctement accessibles.

---

### Phase 2 : Conception de la Fonction de Récompense (*Reward Shaping*)

Un apprentissage avec récompenses clairsemées (*sparse rewards*) est généralement inefficace en robotique. Il est donc nécessaire de définir une fonction de récompense dense et progressive.

- **Récompense de proximité**  
Encourager l’effecteur terminal à se rapprocher de la cible (ex. poignée de porte).
- **Récompense d’interaction**  
Valoriser les contacts pertinents :
  - toucher l’objet ;
  - saisir correctement avec le gripper.
- **Récompense de progression**  
Mesurer l’avancement vers l’objectif :
  - angle d’ouverture d’une porte ;
  - position cible atteinte ;
  - état final de l’objet.
- **Récompense terminale**  
Bonus significatif en cas de réussite complète de la tâche.

---

### Phase 3 : Apprentissage Progressif (*Curriculum Learning*)

Pour stabiliser l’apprentissage, introduire progressivement la difficulté :

- **Étape 1 : Configuration simplifiée**  
Initialiser le robot très proche de la cible.
- **Étape 2 : Augmentation de la difficulté**  
Élargir progressivement la distribution des positions initiales.
- **Étape 3 : Variabilité environnementale**  
Introduire des variations :
  - position/orientation des objets ;
  - légères perturbations (si pertinent).

Objectif : améliorer la robustesse et la généralisation de la politique.

---

### Phase 4 : Combinaison Imitation + Renforcement (IL + RL)

Exploiter les données disponibles pour accélérer l’apprentissage.

- **Pré-entraînement par imitation (*Behavior Cloning*)**  
Utiliser des démonstrations pour initialiser la politique :
  - apprentissage supervisé des actions ;
  - convergence rapide vers un comportement plausible.
- **Affinage par apprentissage par renforcement (RL)**  
Optimiser la politique dans l’environnement simulé avec :
  - PPO (*Proximal Policy Optimization*) ou
  - SAC (*Soft Actor-Critic*).

👉 Cette combinaison permet de réduire drastiquement le temps d’exploration.

---

### Phase 5 : Évaluation et Livrables

- **Code**
  - Archive contenant :
    - modèle entraîné ;
    - scripts d’entraînement et d’évaluation ;
    - README détaillant la reproduction des résultats.
- **Rapport**  
Inclure :
  - description de la méthode ;
  - choix de conception (MDP, reward, etc.) ;
  - difficultés rencontrées et solutions ;
  - courbes d’apprentissage ;
  - visualisations / démonstrations de la tâche.

---

## Document de Conception (à formaliser avant implémentation)

Avant de coder, définir précisément le problème sous forme de **MDP (Markov Decision Process)** :

1. **Espace d’état (State Space)**  
Informations disponibles :
  - positions articulaires ;
  - pose de l’effecteur ;
  - état des objets.
2. **Espace d’action (Action Space)**  
Type de contrôle :
  - couples articulaires (bas niveau) ;
  - contrôle cartésien (plus simple pour débuter).
3. **Fonction de récompense (Reward Function)**  
Formulation mathématique complète (pondérations incluses).
4. **Conditions de terminaison**
  - succès ;
  - limite de temps ;
  - conditions d’échec (collision, instabilité, etc.).


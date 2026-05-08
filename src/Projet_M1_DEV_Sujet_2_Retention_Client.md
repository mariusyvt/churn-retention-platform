# Projet – Sujet 2 - Système Intelligent Multi-Modèles pour la Rétention Client et l'Évaluation du Risque de Revenus

**EFREI, M1 Dev. Manager Full Stack — Sarah Malaeb — 2025-26**
**Data Science**

> Vous devez soumettre vos livrables sur MOODLE, en respectant la date limite fixée par votre enseignant.
> Soumettez votre travail personnel. Toute tricherie ne sera pas tolérée et sera sanctionnée.
> Ce projet peut être réalisé en groupe de quatre membres.

---

## Cadre Théorique et Rôle du Projet

**Machine Learning Supervisé, Deep Learning, Dashboarding et API**

### Machine Learning Supervisé : Fondements, Méthodes et Rôle dans le Projet

Le Machine Learning supervisé constitue l'un des piliers fondamentaux de l'intelligence artificielle appliquée. Il repose sur un principe simple mais puissant : apprendre automatiquement une fonction de prédiction à partir de données d'entrée (features) associées à une variable cible connue (label). Contrairement à une approche purement statistique descriptive, l'objectif n'est pas seulement de comprendre le passé, mais de généraliser à des situations futures jamais observées.

Dans un contexte business et marketing, cette capacité prédictive devient stratégique. Les entreprises SaaS, télécom ou e-commerce collectent en permanence des données relatives à leurs clients : données démographiques, fréquence d'utilisation du service, durée des sessions, historique de paiement, interactions avec le support client, satisfaction (NPS), tickets ouverts, remises appliquées, etc.

Ces informations contiennent des signaux comportementaux parfois subtils pouvant annoncer une probabilité élevée de résiliation (churn) ou une baisse significative de revenu.

Le Machine Learning supervisé permet d'exploiter ces données pour :

- Prédire la probabilité qu'un client résilie son abonnement (classification binaire),
- Identifier des segments de clients à risque (classification),
- Estimer le revenu à risque associé à un client (régression),
- Prédire la valeur vie client (Customer Lifetime Value CLV) (régression).

Ainsi, le modèle ne se contente pas d'analyser le passé ; il devient un outil stratégique d'aide à la décision permettant de prioriser les actions de fidélisation et d'optimiser les campagnes de rétention.

Contrairement à une simple corrélation statistique, le Machine Learning :

- apprend automatiquement des relations complexes entre comportements clients et churn,
- capture des interactions non linéaires invisibles à l'œil humain,
- s'adapte à des profils clients hétérogènes,
- permet une prise de décision automatisée et scalable à grande échelle.

Dans ce projet, vous ne devez pas vous limiter à un seul algorithme. Un des objectifs pédagogiques majeurs est de comprendre que chaque modèle possède ses hypothèses, ses forces et ses limites. Par exemple :

- Une **régression logistique** peut être robuste et interprétable,
- Un **Random Forest** peut capturer des comportements non linéaires,
- Un **Gradient Boosting** peut offrir de meilleures performances prédictives,
- Un **réseau neuronal** peut modéliser des interactions complexes entre variables comportementales.

Vous devrez comparer ces approches de manière rigoureuse afin d'identifier :

- Le modèle le plus performant,
- Le modèle le plus stable,
- Le modèle le plus interprétable,
- Le meilleur compromis performance / complexité dans un contexte business.

L'objectif n'est pas uniquement d'obtenir le meilleur score, mais de comprendre **pourquoi** un modèle fonctionne mieux qu'un autre dans un contexte de rétention client.

---

### Deep Learning : Rôle, Intérêt et Analyse Critique

Le Deep Learning représente une évolution du Machine Learning classique, reposant sur des réseaux de neurones artificiels multicouches capables d'apprendre des représentations hiérarchiques des données.

Dans un environnement industriel, les relations entre capteurs et événements de panne peuvent être hautement non linéaires et dépendre de combinaisons complexes de signaux. Les réseaux de neurones (MLP — Multi Layer Perceptron) permettent de capturer ces interactions sans nécessiter une ingénierie de features excessive.

Dans ce projet, vous pouvez intégrer :

- Un modèle Deep Learning tel que le MLP pour une tâche de classification, ou
- Un modèle Deep Learning pour une tâche de régression.

Cependant, l'intégration du Deep Learning ne doit pas être automatique ou dogmatique. Elle doit être **justifiée**. Vous devrez analyser :

- Quand un modèle simple est suffisant,
- Quand un modèle plus complexe apporte un gain réel,
- Le compromis biais / variance,
- Le risque d'overfitting,
- Le coût computationnel.

Un réseau neuronal mal paramétré peut sur-apprendre et devenir instable. À l'inverse, un modèle trop simple peut sous-apprendre et ne pas capturer les dynamiques clients.

L'objectif pédagogique est donc de :

1. Comprendre que le Deep Learning n'est pas toujours supérieur.
2. Apprendre à comparer scientifiquement les performances dans un contexte marketing réel.

---

### Dashboarding et Data Visualization : De la Prédiction à la Décision

Un modèle performant mais incompréhensible n'a que peu de valeur en entreprise. Dans un contexte business réel, les décideurs (CMO, responsables marketing, responsables CRM, direction financière) ont besoin d'outils visuels clairs et interactifs.

C'est pourquoi ce projet intègre obligatoirement la conception d'un **dashboard décisionnel**.

Le dashboard n'est pas un simple affichage graphique. Il doit permettre :

- La visualisation des distributions des profils clients,
- L'analyse des facteurs influençant le churn,
- La comparaison des performances des modèles,
- La simulation d'un scénario client (ex. augmentation de la fréquence d'utilisation),
- L'obtention d'une probabilité de churn en temps réel,
- L'estimation du revenu à risque,
- L'analyse des variables les plus influentes dans la décision,
- Etc.

Vous devrez adopter une approche orientée **utilisateur métier**. Posez-vous les questions suivantes :

- Si j'étais responsable marketing, quelles informations seraient prioritaires ?
- Comment visualiser clairement le revenu à risque global ?
- Comment prioriser les clients à contacter ?
- Comment expliquer pourquoi un client est classé à haut risque ?

Le développement de l'interface pourra s'appuyer sur **Streamlit**, **Dash** ou tout autre outil jugé pertinent. L'objectif est de transformer un modèle académique en outil stratégique de pilotage de la rétention.

Dans le monde professionnel, un modèle de Machine Learning n'est pas utilisé directement dans un notebook. Il peut être intégré dans une architecture logicielle plus large : CRM, plateforme marketing automation, application interne, etc. Cette intégration passe par la mise en place d'une **API REST**.

Une API permet à d'autres systèmes d'interagir avec votre modèle. Par exemple :

- Un CRM pour scorer automatiquement les clients,
- Une plateforme d'emailing pour cibler les campagnes,
- Un système décisionnel interne.

Dans ce projet, vous devrez développer une API REST (via **FastAPI** ou **Flask** ou autre) comprenant :

- Un endpoint `/predict` recevant les données client,
- Un endpoint `/health` vérifiant l'état du service,
- Optionnel : `/model-info` fournissant des informations sur le modèle.

L'objectif pédagogique est d'introduire la notion d'**industrialisation** :

- Séparation modèle / interface,
- Sérialisation du modèle,
- Gestion des erreurs,
- Structuration d'un service IA.

Vous devez comprendre qu'un modèle performant n'est qu'une partie du système. Ce qui crée de la valeur en entreprise, c'est son intégration dans un pipeline complet.

---

### Rôle Global du Projet

Ce projet ne consiste pas uniquement à entraîner un modèle. Il s'agit de concevoir un **système intelligent complet**, comprenant :

1. Préparation des données clients,
2. Modélisation multi-algorithmes,
3. Évaluation comparative,
4. Interprétabilité,
5. Interface utilisateur décisionnelle,
6. API déployable.

Vous adoptez ainsi une posture de développeur/consultant IA, capable de passer des données brutes clients à une plateforme stratégique d'aide à la décision pour la rétention et l'optimisation du revenu.

---

## Objectifs Pédagogiques du Projet

En réalisant ce projet, vous apprendrez à :

- **Analyser et préparer les données** : réaliser un audit complet du dataset (valeurs manquantes, outliers, incohérences, fuites de données), définir une stratégie de nettoyage et concevoir un feature engineering pertinent adapté à la problématique métier.
- **Implémenter des modèles d'apprentissage supervisé** : développer et entraîner plusieurs algorithmes de Machine Learning (et éventuellement Deep Learning), en comprenant leurs hypothèses, leurs limites et leurs domaines d'application.
- **Évaluer rigoureusement les performances des modèles** : sélectionner et interpréter des métriques adaptées au problème (RMSE, R² pour la régression ; Accuracy, Precision, Recall, F1-score, ROC-AUC pour la classification), et mettre en œuvre des stratégies de validation robustes (validation croisée, analyse des erreurs).
- **Interpréter les modèles et expliquer les résultats** : analyser les contributions des variables (feature importance, SHAP ou équivalent) afin de rendre les modèles compréhensibles et exploitables dans un contexte métier.
- **Comparer et sélectionner un modèle optimal** : comparer plusieurs modèles selon des critères de performance, robustesse, complexité et interprétabilité, et justifier le choix du modèle final en lien avec les contraintes métier.
- **Concevoir un pipeline Data Science complet** : structurer un workflow de bout en bout intégrant la préparation des données, la modélisation, l'évaluation et l'exploitation des résultats dans une logique industrialisable.
- **Assurer la reproductibilité et la traçabilité des expérimentations** : mettre en place des scripts d'exécution (CLI/Makefile), gérer les environnements (`requirements.txt` / `pyproject.toml`), versionner les artefacts (modèles, métriques, figures) et suivre les expérimentations (logs, fichiers CSV, MLflow ou équivalent).
- **Déployer un Proof of Concept (POC) opérationnel** : exposer le modèle via une interface exploitable : script CLI (`predict.py`), notebook interactif ou micro-API (FastAPI/Flask) avec un endpoint de prédiction.
- **Communiquer les résultats de manière professionnelle** : rédiger un rapport synthétique (≤ 6 pages) présentant les données, la méthodologie, les résultats, les limites, les biais et des recommandations actionnables.
- **Adopter une démarche projet collaborative** : travailler en équipe dans une logique d'ingénierie (organisation, répartition des tâches, versioning, qualité du code).
- **Se positionner comme un consultant en Data Science / IA** : présenter une solution de manière claire, structurée et orientée décision, en mettant en avant la valeur métier des résultats.

---

## Compétences visées

### Compétences principales évaluées

- Collecter et préparer des données industrielles
- Concevoir et entraîner plusieurs modèles ML/DL
- Évaluer et comparer des performances
- Prototyper une solution IA complète
- Développer une API d'inférence
- Concevoir un dashboard décisionnel
- Documenter et justifier les choix techniques

### Compétences transverses

- Travail collaboratif
- Rigueur méthodologique
- Capacité d'analyse critique
- Présentation professionnelle

---

## Grille de notation (non certifiante)

*(rf. Syllabus du module)*

| Domaine | Critères | Points |
|---|---|---|
| Préparation et qualité des données | Audit, nettoyage, features | 4,0 |
| Modèles et évaluation | Baseline, métriques, CV, robustesse | 5,0 |
| Reproductibilité & traçabilité | Scripts, env, artefacts, expériences | 3,0 |
| Exposition POC | CLI/notebook/app, tests de bon fonctionnement | 3,0 |
| Rapport et visualisations | Clarté, limites, biais, reco | 3,0 |
| Qualité logicielle | Structure, tests, doc | 2,0 |
| **Total** | | **20,0** |

> *NB : évaluation pédagogique (non certifiante), destinée à vérifier l'atteinte des acquis du module.*

---

## Cahier des Charges du Projet

### Description du projet

Dans le cadre de ce projet, vous devez concevoir et développer une **plateforme intelligente de rétention client** capable d'exploiter des données issues d'un environnement business (SaaS, télécom, services par abonnement) afin d'anticiper le risque de résiliation (churn) et d'évaluer l'impact financier associé.

Le dataset utilisé simule un environnement business réaliste dans lequel différents profils clients génèrent en continu des données comportementales telles que la fréquence d'utilisation du service, la durée des sessions, l'historique de facturation, les incidents de paiement, les interactions avec le support client, les scores de satisfaction (CSAT, NPS), ou encore le type de contrat.

L'objectif est de transformer ces données clients en un **système d'aide à la décision** capable de détecter des patterns annonciateurs de résiliation ou de baisse de revenu. Le système que vous développerez pourra être capable d'effectuer plusieurs tâches prédictives complémentaires :

- Prédire la probabilité qu'un client résilie son abonnement (classification binaire),
- Identifier des segments clients à haut risque (classification multi-classe ou segmentation supervisée),
- Estimer le revenu à risque associé à un client (régression),
- Prédire la valeur vie client (Customer Lifetime Value – CLV) (régression),
- Etc.

Cependant, dans le cadre de ce projet pédagogique, **vous n'êtes pas tenus de réaliser plusieurs tâches de prédiction**. Vous devez choisir **une seule tâche de prédiction** de votre choix (classification ou régression) et construire une solution complète et rigoureuse autour de celle-ci.

> **Points Bonus** : Pour plusieurs tâches de prédiction.

---

### Objectif

L'objectif de ce projet est de concevoir un **MVP (Minimum Viable Product)** professionnel intégrant l'ensemble des briques fondamentales d'un système d'intelligence artificielle business.

Vous devrez combiner plusieurs dimensions complémentaires :

- **Une dimension Data Science** : nettoyage, préparation des données, modélisation supervisée et évaluation rigoureuse des performances.
- **Une dimension comparative** : implémentation de plusieurs algorithmes de ML et de DL (optionnel) afin d'analyser leurs performances respectives.
- **Une dimension décisionnelle** : développement d'un dashboard interactif permettant d'exploiter les prédictions de manière claire et exploitable par un utilisateur métier.
- **Une dimension d'industrialisation** : mise en place d'une API REST permettant d'exposer le modèle sous forme de service.

---

## Dataset

Vous pouvez trouver le dataset sur Kaggle :
[https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset)

Il s'agit du fichier `customer_churn.csv`.

**Caractéristiques :**

- 10 000 enregistrements (clients)
- Variables numériques et catégorielles
- Variable cible principale : `churn` (0 = No, 1 = Yes)
- Données synthétiques mais générées selon une logique métier réaliste
- Corrélations cohérentes entre comportements clients et probabilité de churn

**Variables clés (exemples) :**

| Variable | Description |
|---|---|
| `age` | Âge du client |
| `gender` | Genre |
| `tenure` | Ancienneté |
| `contract_type` | Type de contrat |
| `monthly_charges` | Charges mensuelles |
| `total_revenue` | Revenu total |
| `payment_failures` | Échecs de paiement |
| `support_tickets` | Tickets support |
| `session_duration` | Durée de session |
| `login_frequency` | Fréquence de connexion |
| `nps_score` | Score NPS |
| `churn` | Variable cible |

---

## Problématiques prédictives possibles

### 1. Prédiction du Churn (Classification Binaire) *(tâche principale)*

**Variable cible :** `churn`

**Intérêt business :**
- Prioriser les actions de rétention
- Réduire le coût d'acquisition
- Stabiliser le chiffre d'affaires

### 2. Estimation du Revenu à Risque (Régression)

**Variable cible :** `revenue_at_risk = total_revenue * proba_churn`  
ou `expected_loss = monthly_fee * probabilité_churn`

**Intérêt business :**
- Priorisation des clients à forte valeur
- Arbitrage budget marketing
- Optimisation des campagnes

### 3. Estimation de la Valeur Vie Client CLV (Régression)

**Variable cible :** `total_revenue`

**Intérêt business :**
- Identifier clients premium
- Adapter les offres
- Segmenter intelligemment

---

## Prérequis et exigences principales

- L'analyse doit inclure **au minimum 3 modèles** différents
- Au moins 1 modèle Deep Learning (optionnel)
- Comparaison quantitative obligatoire
- Dashboard interactif obligatoire
- API fonctionnelle
- Interprétation de l'importance de features
- Interprétation du modèle
- Séparation train/test rigoureuse
- Cross-validation recommandée

---

## Résultats attendus

- Comparaison détaillée des performances
- Choix justifié du meilleur modèle
- Dashboard fonctionnel
- API accessible
- Rapport analytique structuré

---

## Recommandations

### Commencez par une analyse exploratoire approfondie

Avant toute modélisation, prenez le temps de comprendre vos données : distributions, valeurs extrêmes, cohérence des variables, présence de valeurs manquantes, éventuelles anomalies et relations entre variables. Une EDA bien menée vous permet de construire une intuition métier.

### Vérifiez les distributions des classes

Dans un problème de churn, il est fréquent que la classe `churn = 1` soit minoritaire. Un modèle peut donc afficher une accuracy élevée tout en étant inefficace pour détecter les clients réellement à risque. Vous devez analyser le déséquilibre des classes et privilégier des métriques adaptées (Recall, F1, PR-AUC), voire appliquer des stratégies appropriées (stratified split, `class_weight`, ajustement du seuil de décision).

### Comment gérer explicitement le déséquilibre des classes ?

Si un déséquilibre significatif est observé, vous devez obligatoirement expérimenter et comparer plusieurs techniques :

**Analyse préalable :**
- Calculez le ratio de déséquilibre
- Analysez la matrice de confusion d'un modèle baseline
- Montrez en quoi l'accuracy est insuffisante dans ce contexte
- Utilisez des métriques adaptées : Recall, F1-score, ROC-AUC, PR-AUC (fortement recommandé si déséquilibre important)

**Techniques de rééquilibrage (data-level) :** implémentez et comparez au moins deux méthodes parmi :
- Random Over-Sampling
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random Under-Sampling

**Stratégie de validation :** utilisez une validation croisée stratifiée (Stratified K-Fold).

**Approches au niveau du modèle :**
- Pondération des classes (`class_weight = "balanced"`)
- Apprentissage sensible au coût (cost-sensitive learning)

**Ajustez le seuil de décision :** testez différents seuils et identifiez un seuil optimisé (ex. maximisation du F1 ou du Recall).

### Analysez la corrélation et la redondance des variables

L'étude des corrélations est importante pour identifier les variables redondantes et limiter certains risques (multicolinéarité). Parfois, une variable dérivée est plus pertinente qu'une variable brute (ex. ratio `support_tickets / tenure_months`, évolution d'usage via `usage_growth_rate`).

### Implémentez d'abord un modèle baseline, puis complexifiez progressivement

Commencez par un modèle simple (régression logistique, régression linéaire) pour établir un point de référence. Ensuite, introduisez progressivement des modèles plus puissants (Random Forest, Gradient Boosting, SVM, MLP, etc.).

### Comparez systématiquement au moins 3 modèles

À l'issue de la comparaison, sélectionnez un **modèle candidat final** argumenté en prenant en compte : performance, stabilité, interprétabilité, coût de calcul, facilité de déploiement et cohérence métier.

### Évitez le data leakage

Les étapes de preprocessing (imputation, scaling, encodage) doivent être ajustées uniquement sur le train set puis appliquées au test set. L'usage de pipelines (`sklearn.Pipeline`, `ColumnTransformer`) est fortement recommandé.

### Utilisez la validation croisée

Lorsque c'est pertinent, utilisez une validation croisée (cross-validation) afin de rendre vos résultats plus robustes et moins dépendants d'un split unique.

### Optimisez les hyperparamètres de manière raisonnable

Vous pouvez utiliser `GridSearch` ou `RandomizedSearch`, en expliquant votre stratégie et en sélectionnant des plages réalistes.

### Analysez les erreurs

Ne vous contentez pas de scores globaux. Analysez les erreurs : matrices de confusion, résidus en régression, exemples de cas mal prédits.

### Travaillez l'interprétabilité

Utilisez des techniques d'importance des variables ou SHAP pour expliquer les décisions du modèle.

Un responsable CRM ou marketing doit pouvoir répondre à des questions telles que :
- Pourquoi ce client est-il classé à haut risque de churn ?
- Quels facteurs expliquent cette probabilité élevée ?
- Est-ce lié au prix, à l'engagement, au support client ou à un problème de paiement ?
- Quelle action concrète peut être mise en place pour réduire ce risque ?

#### Quand appliquer les techniques d'explicabilité (Feature Importance, SHAP) ?

Les techniques d'explicabilité s'appliquent **après l'entraînement du modèle**, lors de la phase d'évaluation et d'interprétation des prédictions.

| Technique | Quand l'utiliser | Niveau |
|---|---|---|
| `feature_importances_` | Après entraînement | Basique |
| Permutation Importance | Après évaluation | Recommandé |
| SHAP | Sur le modèle final sélectionné | Avancé |

**Feature Importance (importance globale) :**
- Importance native des modèles basés sur les arbres (Random Forest, Gradient Boosting, XGBoost, LightGBM…) : `model.feature_importances_`
- Permutation Importance (recommandée) : `from sklearn.inspection import permutation_importance`

**SHAP (explicabilité locale et globale) :**
SHAP permet de :
- Expliquer une prédiction individuelle (vision locale)
- Obtenir une importance globale des variables
- Identifier l'impact positif ou négatif de chaque variable sur une prédiction donnée

### Structurez proprement votre code

Organisez votre projet en modules (data preprocessing, modeling, evaluation, API, dashboard). Évitez un notebook monolithique désorganisé. Séparez les responsabilités et assurez la reproductibilité.

### Versionnez votre travail régulièrement avec Git

Effectuez des commits réguliers et explicites. Un dépôt Git propre et vivant constitue un indicateur fort de maturité professionnelle.

### Dashboard : indépendant, exploitable, orienté décision

Votre dashboard doit être conçu comme un **outil décisionnel autonome**. Il doit permettre de visualiser les données, d'explorer des indicateurs clés, de comparer les modèles, et d'exécuter des prédictions sur des scénarios saisis par l'utilisateur.

### API : testez indépendamment du dashboard

Vérifiez vos endpoints dès le début avec Postman, curl, ou un script Python. Assurez-vous que `/predict` renvoie exactement ce qui est attendu, avec une gestion robuste des erreurs (champs manquants, types incorrects, valeurs incohérentes).

L'API devra inclure au minimum :
- `POST /predict` : reçoit un JSON contenant les features et renvoie la prédiction
- `GET /health` : endpoint de santé vérifiant que le service est actif
- Gestion des erreurs : validation des entrées, message d'erreur clair et code HTTP approprié
- Documentation minimale : README ou Swagger/FastAPI

> **Important :** le dashboard devra idéalement appeler l'API pour obtenir les prédictions (et non charger directement le modèle), afin de reproduire une architecture réaliste (Front / API / Modèle).

---

## Exigences Fonctionnelles Détaillées (EF)

### EF1 : Acquisition et Préparation des Données

Mettre en place un pipeline de préparation incluant : nettoyage des valeurs manquantes, encodage des variables catégorielles, normalisation/standardisation si nécessaire, et analyse exploratoire documentée.

### EF2 : Modélisation Multi-Algorithmes

Entraîner et comparer **au minimum 4 modèles** (classification et/ou régression selon votre choix), incluant des modèles de référence et des modèles plus avancés. Sélectionner et justifier un modèle candidat final.

### EF3 : Système d'Évaluation

Utiliser des métriques adaptées (classification : Accuracy, Precision, Recall, F1, ROC-AUC ; régression : MAE, RMSE, R²…) et produire des comparatifs (tableaux et graphes) ainsi qu'une analyse d'erreurs.

### EF4 : Dashboard Interactif

L'interface doit permettre la saisie d'un scénario, afficher la prédiction, comparer les modèles, montrer l'importance des variables et intégrer des graphiques interactifs. Framework conseillé : **Streamlit** (ou Dash/Plotly) ou tout autre outil jugé pertinent.

### EF5 : API

Développer une API REST exposant au minimum : `POST /predict`, `GET /health`, gestion des erreurs, et documentation minimale (README ou Swagger/FastAPI).

---

## Livrables

Vous devez soumettre les livrables sur **MOODLE** et **Git/GitHub** :

- Code source de la solution fonctionnelle (+ documentation technique en annexe)
- Rapport du projet (6 pages)
- Support de Présentation du projet
- Vidéo de démonstration *(uniquement si vous ne pouvez pas présenter en classe lors de la dernière séance du module)*

### Consignes pour la présentation

Une présentation avec démonstration en classe est prévue lors de la dernière séance du module.

- Tous les membres du groupe doivent participer activement.
- L'évaluation sera **individuelle**.
- En cas d'impossibilité de présenter en classe, vous devrez soumettre une vidéo de présentation et de démonstration complète.

---

**GOOD LUCK! 🍀**

#Rapport- Projet NLP Text Mining#

##Auteurs : Romain BUONO, Yassine CHENIOUR, Miléna GORDIEN-PIQUET, Anne-Camille VIAL##

##Introduction##

Le présent rapport décrit l’ensemble des procédures, des stratégies méthodologiques et des dispositifs techniques mis en œuvre par le groupe n°6 de la promotion 2025-2026 du Master 2 SISE,composé des auteurs du rapport, dans le cadre du projet de NLP et de text mining réalisé sous la supervision de M. Ricco RAKOTOMALALA.
Le projet s’articule autour de plusieurs objectifs principaux : 
(i) la sélection et la constitution d’un corpus issu de plateformes en ligne de diffusion d’offres d’emploi ; 
(ii) la structuration et le stockage de ce corpus au sein d’un entrepôt de données reposant sur un système de gestion de bases de données libre ; 
et (iii) le développement d’une application connectée à cet entrepôt, destinée à l’analyse multidimensionnelle des annonces d’emploi, avec une attention particulière portée à la dimension géographique.
Afin d’en faciliter la lecture, le document adopte une organisation chronologique, fidèle aux différentes étapes du projet. La première section est consacrée à la justification du choix du corpus ainsi qu’aux modalités de son acquisition. 
La deuxième section traite des enjeux liés à la conception et à l’alimentation de l’entrepôt de données. La troisième section présente l’architecture de l’application développée, en mettant en lumière les problématiques spécifiques rencontrées lors de sa conception et de son implémentation. 
Enfin, une dernière section synthétise les principaux résultats obtenus, tant sur le plan technique que fonctionnel.


##Résumé##
Le projet détaillé dans le présent rapport est développé dans le cadre du cours "NLP-Text Mining" dispensé par Ricco RAKOTOMALALA.


 ##Section 1 : Corpus documentaire, Base NoSQL sous MongoDB puis Data WhareHouse sous MotherDuckDB##
 
   Afin d’assurer la cohérence avec la thématique de la formation, le périmètre de l’étude a été volontairement restreint aux métiers relevant des domaines de la data et de l’intelligence artificielle, dans leur dimension d'exploitation technique. Ainsi les métiers liés à ces technologies mais ne nécessitant pas une manipulation et/ou exploitation de celles-ci n'ont pas été retenus (par ex : commercial, ressources humaines, etc.).

L’objectif général du projet étant de produire une représentation aussi fidèle et exhaustive que possible du marché professionnel concerné, il est apparu indispensable de s’appuyer sur des sources multiples et hétérogènes, afin de constituer un échantillon d’offres d’emploi suffisamment large et diversifié.
Dans cette perspective, chaque auteur s’est vu confier l’exploitation d’une source spécifique et a développé une méthode de scraping adaptée aux particularités de celle-ci, tout en contribuant à l’alimentation automatisée d’une base commune NoSQL.
La suite de cette analyse vise à expliciter les raisons pour lesquelles le recours à la technologie NoSQL s’est imposé face à la diversité des sources mobilisées, avant de présenter l’architecture retenue ainsi que les enjeux techniques associés à chacune des méthodes de scraping mises en œuvre.


###Sous section 1 : Le choix de l'alimentation d'une base de données NoSQL sous MongoDB###
Pour rappel, les consignes initiales du projet prévoient que les données relatives au marché de l'emploi doivent être stockées et structurées dans un data wharehouse hébergé sur un SGBD libre. Bien que des solutions technologiques existent pour satisfaire à ces critères, la plupart, si ce n'est la totalité d'entre elles, imposent une limite maximale quant au volume de données stocké.
Cette limite peut s'avérer être un frein important dans les débuts du projet, attendu qu'il peut être plus qu'hasardeux de prédire le volume effectivement mobilisé par des données scrapées de sources diverses.
En outre, la structuration d'un data wharehouse, bien qu'elle se doive de répondre aux besoins métiers, est également conditionnée par le type et l'architecture des données qui l'alimentent.
Il est donc apparu dans ce contexte nécessaire de passer par une étape intermédiaire de stockage des données brutes, afin de pouvoir conserver une certaine souplesse dans les méthodes employées dans les phases initiales du projet et donner l'opportunité d'observer les données avant de se conformer à une structure posant le risque de ne pas correspondre à la réalité des données.
La caractéristique fondamentale d’une base de données No SQL réside dans la flexibilité de son schéma :  son adaptabilité aux différentes structures possibles d’un même set de données résultant des méthodes de scraping et à l’hétérogénéité des structures des différents documents. 
Ce format s'est donc imposé afin de faire cohabiter initialement les documents issus des procédures de scraping, sans avoir à définir de formalisme rigide.
Le choix de l'hébergement de cette  base de données initiale s'est porté sur MongoDB en raison de ses fonctionnalités.
En effet , la plateforme MongoDB est particulièrement adaptée au stockage de documents en format .json, attendu que le stockage est réalisé en BJSON (Binary JSON). Ce choix technologique permet de stocker directement les productions des procédures de scraping sans aucune transformation nécessaire préalable. Par ailleurs, le JSON permettant d’obtenir des listes ou bien des objets dans des objets, MongoDB gère nativement ces hiérarchies. Cette fonctionnalité permet de conserver la structure des éléments scrapés sans risque de perte d’information.


###Sous section 2 : 4 procédures de Scraping
 
 ####- France Travail
         Le site francetravail.fr propose une API officielle d'extraction des annonces de sa base de données.L'extraction de données à grande échelle à partir d'API publiques présente des défis structurels : latence réseau, limitations de débit (rate limiting) et hétérogénéité des données. Ce projet expose la conception d'un système de scraping "production-grade" capable d'automatiser la collecte d'offres d'emploi liées à l'Intelligence Artificielle, en optimisant le compromis entre performance et conformité aux politiques d'utilisation de l'API.
         
 Le système est conçu selon une architecture de pipeline à trois niveaux, garantissant une séparation stricte entre l'authentification, la découverte et l'extraction.
 
 **Pipeline de traitement :**
 
•	Étage 1 (Authentification) : Implémentation du protocole OAuth2 pour la gestion des jetons d'accès.
•	Étage 2 (Moteur de Recherche) : Pagination récursive et dédoublonnage par hachage avec une complexité temporelle de $O(1)$.
•	Étage 3 (Scraper Parallèle) : Unité d'extraction granulaire gérant le filtrage par expressions régulières (RegEx).

**Implémentation Technique**
1. Concurrence et Gestion du Débit
Pour pallier l'inefficacité du traitement séquentiel, nous avons implémenté un ThreadPoolExecutor. Le choix des fils d'exécution (threads) plutôt que des processus se justifie par la nature "I/O-bound" de la tâche, minimisant ainsi l'empreinte mémoire.
La conformité avec l'API est assurée par un limiteur de débit basé sur l'algorithme de la fenêtre glissante (Sliding Window). Le débit maximum ($T$) est défini par :
$$T_{max} = \min \left( N_{workers} \cdot \frac{1}{L_{api}}, \text{RateLimit} \right)$$
Où $L_{api}$ représente la latence réseau moyenne.
2. Ingénierie des Données
Le stockage s'appuie sur des dataclasses Python, assurant un typage fort et une sérialisation efficace vers le format JSON. Le filtrage sémantique utilise des frontières de mots (\b) pour garantir une précision accrue lors de l'identification des compétences clés (Data, IA, Machine Learning).

**Analyse des Performances et Résultats**
L'approche parallèle a démontré une amélioration substantielle du débit de données par rapport à l'approche séquentielle de référence.
Tableau 1 : Comparatif des performances
| Métrique | Séquentiel | Parallèle (8 workers) | Gain |
| :--- | :--- | :--- | :--- |
| Temps (100 requêtes) | 32.5 s | 11.1 s | +192% |
| Débit (offres/sec) | 3.1 | 9.0 | 2.9× |
| Taux de succès | 95%+ | 95%+ | Constant |
Analyse de la Qualité (Précision)
Le filtrage post-extraction a permis d'affiner les résultats avec une précision de 63.6%, éliminant les faux positifs générés par le moteur de recherche natif de l'API.

**Conclusion**
Le système développé démontre que l'utilisation de la programmation concurrente, couplée à un contrôle rigoureux du débit, permet de construire des pipelines d'acquisition de données robustes et scalables. Cette architecture est directement transposable à d'autres plateformes de données ouvertes (Open Data), constituant une base solide pour des analyses ultérieures du marché de l'emploi.

 ####- APEC
 ####- Jobteaser
 ####- Choisir le service public

 ##Section 2 : Entrepôt de données, sous MotherDuck ##
  - uniformisation des variables de chaque collection du datalake
  - Choix de technologie MotherDuck, issu de DuckDB
  - architecture de l'entrepôt
  - script d'alimentation

 ##Section 3 : Application Streamlit##
  - rappel des contraintes imposées de l'application
  - architecture de l'application
  - choix des analyses
  - structuration et fonctionnalités de la partie cartographique

##Conclusions##
 - conclusions techniques  : parallélisation, choix de technologies...
 - conclusions fonctionnelles : localisation et statistiques des KPI

##Annexes##

##Références##
 

 

 

 

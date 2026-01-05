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
 Reprise des principaux éléments du projet 

 ##Section 1 : Corpus documentaire, Datalake sous MongoDB##
   - quatre méthodes de scraping, pour 4 sources distinctes
   - alimentation d'un datalake, rendu nécessaire par les sources distinctes

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
 

 

 

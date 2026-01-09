# Moteur de Recherche Sémantique - Architecture et Méthodologie

## Contexte du Module

Ce module implémente un **moteur de recherche sémantique** pour l'application RUCHE (Recherche Unifiée de Carrières et d'Hébergement d'Emplois). L'objectif est de permettre aux utilisateurs de rechercher des offres d'emploi en **langage naturel** plutôt que par mots-clés stricts.

**Problématique** : Les recherches traditionnelles par mots-clés (ex: "Data Engineer" AND "Lyon") sont **rigides** et génèrent :
- **Faux négatifs** : Une offre "Ingénieur Données à Lyon" ne sera pas trouvée (synonyme non détecté)
- **Faux positifs** : Une offre "Data Analyst Paris" pourrait matcher "Data" même si la localisation diffère

**Solution proposée** : Un système de **recherche vectorielle** combinant :
1. **Embeddings sémantiques** : Représentation vectorielle dense capturant le sens des textes
2. **Similarité cosinus** : Mesure de proximité sémantique dans l'espace vectoriel
3. **Compute pushdown** : Calcul côté SQL pour optimiser les performances

---

## Requête "Fil Rouge"

Tout au long de cette fiche, nous suivrons le traitement de cette requête utilisateur :

**Recherche** : _"Data Engineer Lyon"_

**Objectif** : Le système doit retourner les offres les plus pertinentes en comprenant :
- Le **rôle** ("Data Engineer" = ingénieur données)
- La **localisation** ("Lyon")
- Les **synonymes** ("Ingénieur Données", "Engineer Data", etc.)

**Offres fictives dans la base** :

| job_id | title | ville | type_contrat | description | Score attendu |
|--------|-------|-------|--------------|-------------|---------------|
| `job_001` | Data Engineer | Lyon | CDI | Construction de pipelines de données... | ⭐ **Très élevé** |
| `job_002` | Ingénieur Données | Lyon | CDI | Développement d'architectures Big Data... | ⭐ **Élevé** (synonyme) |
| `job_003` | Data Analyst | Paris | CDI | Analyse de données business... | 🔸 **Moyen** (rôle proche, lieu différent) |
| `job_004` | Boulanger | Lyon | CDI | Fabrication de pain et viennoiseries... | ❌ **Très faible** (aucun lien sémantique) |

---

# Phase 1 : Enrichissement Sémantique (ETL Offline)

## Explication Méthodologique

La première phase consiste à **générer des embeddings** (vecteurs sémantiques) pour chaque offre d'emploi. Contrairement à une approche naïve qui vectoriserait uniquement la description brute, nous adoptons une stratégie d'**enrichissement contextuel** via des jointures SQL.

**Pourquoi enrichir le contexte ?**

Un embedding généré uniquement sur la description textuelle **perd des informations cruciales** :
- Le **type de contrat** (CDI vs Stage) influence la nature du poste
- La **localisation** (Paris vs Lyon) est un critère discriminant
- Le **titre** et l'**entreprise** apportent un contexte sémantique fort

Le modèle doit comprendre le **contexte complet** de l'offre pour générer un embedding pertinent.

---

## Architecture Star Schema

Notre base de données suit un **schéma en étoile** (Star Schema) optimisé pour l'analytique :

```
         d_localisation              d_contrat
         ├─ id_ville (PK)            ├─ id_contrat (PK)
         ├─ ville                    └─ type_contrat
         └─ code_postal
                │                          │
                └──────────┐    ┌──────────┘
                           ↓    ↓
                        f_offre (Fact Table)
                        ├─ job_id (PK)
                        ├─ title
                        ├─ description
                        ├─ company_name
                        ├─ hard_skills
                        ├─ soft_skills
                        ├─ id_ville (FK) ──→ d_localisation
                        ├─ id_contrat (FK) ──→ d_contrat
                        └─ embedding FLOAT[768]  ← À peupler
```

**Avantages du Star Schema** :
1. **Normalisation** : Évite la redondance (ville stockée une seule fois)
2. **Jointures simples** : Pas de jointures en cascade (snowflake)
3. **Performance** : Optimisé pour les requêtes analytiques (DuckDB)

---

## Stratégie de Jointure SQL

Pour générer un **document contexte enrichi**, nous effectuons des `LEFT JOIN` entre la table de fait et les dimensions :

```sql
SELECT 
    f.job_id,
    -- Concaténation des champs pour enrichissement sémantique
    CONCAT_WS(' | ',
        f.title,                          -- "Data Engineer"
        f.company_name,                   -- "DataCorp"
        COALESCE(c.type_contrat, ''),    -- "CDI"
        COALESCE(l.ville, ''),            -- "Lyon"
        COALESCE(l.code_postal, ''),      -- "69001"
        f.description,                    -- "Construction de pipelines..."
        COALESCE(f.hard_skills, ''),      -- "Python, SQL, Spark"
        COALESCE(f.soft_skills, '')       -- "Travail en équipe"
    ) AS enriched_text
FROM f_offre f
LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
LEFT JOIN d_contrat c ON f.id_contrat = c.id_contrat
WHERE f.description IS NOT NULL
```

**Pourquoi ces `LEFT JOIN` ?**

1. **`LEFT JOIN d_localisation`** :
   - Récupère la ville en clair ("Lyon") au lieu de l'ID technique (42)
   - Le modèle comprendra mieux "Lyon" que "42"
   - `COALESCE(..., '')` gère les valeurs `NULL` (offres sans localisation)

2. **`LEFT JOIN d_contrat`** :
   - Récupère le type de contrat ("CDI", "Stage", "Alternance")
   - Ces termes ont une **charge sémantique forte** : "Stage Data Engineer" ≠ "CDI Data Engineer"

**Séparateur `CONCAT_WS(' | ', ...)`** :
- `WS` = "With Separator" (séparateur personnalisé)
- Le séparateur `|` évite l'ambiguïté entre champs adjacents
- Exemple sans séparateur : `"DataCorpCDILyon"` (illisible)
- Avec séparateur : `"DataCorp | CDI | Lyon"` (clair)

---

## Application au Fil Rouge

Pour l'offre **`job_001` (Data Engineer à Lyon)** :

### Données brutes (tables séparées)

```sql
-- Table f_offre
job_id: "job_001"
title: "Data Engineer"
company_name: "DataCorp"
description: "Construction de pipelines de données pour..."
hard_skills: "Python, SQL, Apache Spark"
id_ville: 42        ← Clé étrangère
id_contrat: 1       ← Clé étrangère

-- Table d_localisation
id_ville: 42
ville: "Lyon"
code_postal: "69001"

-- Table d_contrat
id_contrat: 1
type_contrat: "CDI"
```

### Document enrichi après jointure

```
enriched_text = "Data Engineer | DataCorp | CDI | Lyon | 69001 | 
                 Construction de pipelines de données pour ingérer, 
                 transformer et stocker des données massives. 
                 Expérience avec Python, SQL, Apache Spark requise. | 
                 Python, SQL, Apache Spark | 
                 Travail en équipe, Autonomie"
```

**Analyse sémantique** :
- **Termes techniques** : "Data Engineer", "pipelines", "données massives", "Python", "Spark"
- **Localisation** : "Lyon", "69001"
- **Contrat** : "CDI"
- **Contexte** : Les mots-clés sont **dispersés** dans le texte, mais le modèle va apprendre les **relations sémantiques** entre eux

---

### Comparaison avec l'offre `job_004` (Boulanger)

```sql
-- Table f_offre
job_id: "job_004"
title: "Boulanger"
company_name: "Boulangerie Artisanale"
description: "Fabrication de pain et viennoiseries traditionnelles..."
hard_skills: "Pétrissage, Cuisson au four"
id_ville: 42        ← Même ville (Lyon)
id_contrat: 1       ← Même contrat (CDI)
```

```
enriched_text = "Boulanger | Boulangerie Artisanale | CDI | Lyon | 69001 | 
                 Fabrication de pain et viennoiseries traditionnelles. 
                 Maîtrise du pétrissage et de la cuisson au four. | 
                 Pétrissage, Cuisson au four | 
                 Rigueur, Passion du métier"
```

**Observation** : Même ville, même contrat, mais **vocabulaire radicalement différent** ("boulanger", "pain", "pétrissage" vs "data", "pipelines", "spark"). Le modèle va capturer cette différence sémantique.

---

# Phase 2 : Choix du Modèle d'Embeddings

## Explication Méthodologique

Le choix du **modèle d'embeddings** est crucial : il détermine la qualité de la représentation vectorielle et donc la pertinence des résultats.

### Modèle Sélectionné

```python
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
```

**Caractéristiques techniques** :
- **Architecture** : MPNet (Masked Permuted Pre-training for Language Understanding)
- **Dimension** : 768 (768 dimensions par vecteur)
- **Multilinguisme** : Entraîné sur 50+ langues dont français et anglais
- **Taille** : ~420 MB (modèle compact)

---

## Justification du Choix

### 1. Multilinguisme (Français + Anglais)

**Contexte RUCHE** : Les offres d'emploi contiennent :
- Du **français** : "Ingénieur Données", "Stage", "Lyon"
- De l'**anglais** : "Data Engineer", "Machine Learning", "DevOps"
- Du **franglais** : "Lead Data Scientist", "Business Intelligence Analyst"

**Problème des modèles monolingues** :
- Un modèle français ne comprend pas "Data Engineer"
- Un modèle anglais ne comprend pas "Ingénieur Données"

**Solution avec modèle multilingue** :
Le modèle a appris que "Data Engineer" (anglais) et "Ingénieur Données" (français) sont **sémantiquement équivalents** grâce à un entraînement sur des corpus parallèles.

**Test empirique** :
```python
model.encode("Data Engineer")
# → [0.234, -0.521, 0.789, ...]

model.encode("Ingénieur Données")
# → [0.231, -0.518, 0.791, ...]  ← Vecteurs très proches !

cosine_similarity(v1, v2) ≈ 0.94  # Haute similarité
```

---

### 2. Dimension 768 : Compromis Performance/Précision

**Théorie** : Plus la dimension est élevée, plus le modèle peut capturer de nuances sémantiques.

**Dimensions courantes** :
| Modèle | Dimension | Précision | Latence |
|--------|-----------|-----------|---------|
| all-MiniLM-L6-v2 | 384 | Bonne | Faible |
| **paraphrase-multilingual-mpnet-base-v2** | **768** | **Très bonne** | **Modérée** |
| all-mpnet-base-v2 | 768 | Excellente | Modérée |
| roberta-large | 1024 | Excellente | Élevée |

**Choix de 768** :
- **Supérieur à 384** : Capture plus de nuances (utile pour distinguer "Data Analyst" vs "Data Engineer")
- **Inférieur à 1024** : Stockage raisonnable (768 × 4 bytes = 3 KB par offre × 5000 offres = 15 MB)
- **Performance** : Encodage de 1000 offres en ~30 secondes (CPU moderne)

---

### 3. Entraînement sur des Paraphrases

Le suffixe `paraphrase-*` indique que le modèle a été **fine-tuné** sur des paires de phrases paraphrasées :

```
"Je cherche un Data Engineer à Lyon"
≈ "Data Engineer basé à Lyon"
≈ "Ingénieur Données région Lyonnaise"
```

**Avantage** : Le modèle comprend que ces phrases expriment la **même intention**, même avec des formulations différentes.

**Application RUCHE** :
```python
# Requête utilisateur
query = "Data Engineer Lyon"

# Offre 1 (formulation proche)
offer_1 = "Data Engineer | DataCorp | CDI | Lyon | ..."
→ Similarité attendue : ~0.85

# Offre 2 (formulation variée)
offer_2 = "Ingénieur Données | TechCorp | CDI | Lyon | ..."
→ Similarité attendue : ~0.78 (légèrement inférieure mais toujours élevée)
```

---

## Application au Fil Rouge

### Encodage du document enrichi (`job_001`)

```python
text = "Data Engineer | DataCorp | CDI | Lyon | 69001 | ..."

embedding = model.encode(text)
# Output: array de shape (768,)
# [0.234, -0.521, 0.789, 0.156, -0.923, ..., 0.412]
```

**Interprétation géométrique** :
- Le vecteur de 768 dimensions place l'offre dans un **espace sémantique**
- Des offres similaires (Data Engineer, Machine Learning Engineer) seront **proches** dans cet espace
- Des offres dissimilaires (Boulanger, Infirmier) seront **éloignées**

---

### Visualisation Conceptuelle (Projection 2D)

En réalité, l'espace est 768D, mais conceptuellement :

```
           Axe "Technique Data/IA"
                    ↑
                    │
          Data      │     ML
        Engineer ●  │  ● Engineer
                    │
     Data Analyst ● │
                    │
────────────────────┼────────────────────→ Axe "Localisation"
                    │                Paris
                  Lyon
                    │
                    │  ● Boulanger
                    │  ● Infirmier
                    ↓
```

**Observation** : "Data Engineer Lyon" sera proche de "Data Analyst Lyon" (même localisation, domaine proche) mais éloigné de "Boulanger Lyon" (localisation identique mais domaine totalement différent).

---

# Phase 3 : Stratégie de Stockage dans MotherDuck

## Explication Méthodologique

Les embeddings générés (768 dimensions par offre) doivent être stockés efficacement pour permettre des recherches rapides. Nous utilisons **MotherDuck** (DuckDB cloud) avec un typage fort.

### Schéma SQL

```sql
ALTER TABLE f_offre 
ADD COLUMN embedding FLOAT[768];
```

**Type `FLOAT[768]`** :
- **`FLOAT`** : Type numérique à virgule flottante (32 bits)
- **`[768]`** : Array de taille fixe (768 éléments)
- **Stockage** : 768 × 4 bytes = **3072 bytes (3 KB)** par offre

---

## Avantages du Pré-calcul (Offline)

### Architecture : Offline vs Online

```
┌─────────────────────────────────────────────────────────────┐
│  OFFLINE (ETL - Une fois)                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Extraction (SQL JOIN)                                  │
│     f_offre + d_localisation + d_contrat                   │
│     ↓                                                       │
│  2. Enrichissement                                          │
│     Document contexte complet                               │
│     ↓                                                       │
│  3. Vectorisation (Sentence Transformer)                   │
│     768D embedding                                          │
│     ↓                                                       │
│  4. Stockage (MotherDuck)                                  │
│     UPDATE f_offre SET embedding = [...]                   │
│                                                             │
│  Durée: ~2 min pour 5000 offres                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ONLINE (Recherche - À chaque requête)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Encodage de la requête utilisateur                     │
│     "Data Engineer Lyon" → embedding 768D                   │
│     Durée: ~50 ms                                          │
│     ↓                                                       │
│  2. Calcul similarité (SQL Server-side)                    │
│     array_cosine_similarity(f.embedding, query_emb)        │
│     Durée: ~100-200 ms (5000 offres)                       │
│     ↓                                                       │
│  3. Tri + TOP 50                                           │
│     ORDER BY similarity DESC LIMIT 50                       │
│                                                             │
│  Durée totale: ~150-250 ms                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Justification** :
1. **Latence utilisateur** : Encodage + recherche < 300 ms (acceptable)
2. **Pas de recalcul** : Les embeddings des offres sont **figés** (mis à jour nuitamment si nouvelles offres)
3. **Scalabilité** : Si 50 000 offres, la latence reste ~1-2s (linéaire en O(n))

---

## Code Python (ETL)

```python
class EmbeddingETL:
    def update_embeddings(self, df: pd.DataFrame):
        """
        Met à jour les embeddings dans MotherDuck
        
        Stratégie:
        1. Créer table temporaire avec embeddings
        2. UPDATE par JOIN (efficace, 1 seule requête)
        3. Nettoyage
        """
        # Créer table temporaire
        self.con.execute(f"""
            CREATE TEMP TABLE temp_embeddings (
                job_id TEXT PRIMARY KEY,
                embedding FLOAT[{self.embedding_dim}]  -- FLOAT[768]
            )
        """)
        
        # Convertir embeddings numpy → liste Python
        df_copy = df.copy()
        df_copy['embedding'] = df_copy['embedding'].apply(lambda x: x.tolist())
        
        # Insertion par batch (1000 offres)
        for i in tqdm(range(0, len(df_copy), 1000)):
            batch = df_copy.iloc[i:i+1000]
            self.con.execute("INSERT INTO temp_embeddings SELECT * FROM batch")
        
        # UPDATE par JOIN (1 seule requête SQL)
        self.con.execute("""
            UPDATE f_offre f
            SET embedding = t.embedding
            FROM temp_embeddings t
            WHERE f.job_id = t.job_id
        """)
        
        # Nettoyage
        self.con.execute("DROP TABLE temp_embeddings")
```

**Avantages de l'UPDATE par JOIN** :
- **1 seule requête** au lieu de 5000 UPDATE individuels
- **Transactionnel** : Soit tout passe, soit rien (atomicité)
- **Performances** : ~10-15s pour 5000 offres (vs plusieurs minutes en UPDATE unitaires)

---

## Application au Fil Rouge

### Stockage de l'embedding `job_001`

```sql
-- Après ETL
SELECT job_id, embedding
FROM f_offre
WHERE job_id = 'job_001';

-- Résultat
job_id: "job_001"
embedding: [0.234, -0.521, 0.789, ..., 0.412]  -- 768 valeurs
```

**Vérification d'intégrité** :
```sql
-- Compter les offres avec embedding
SELECT COUNT(*) FROM f_offre WHERE embedding IS NOT NULL;
-- Output: 4377 (toutes les offres Data/IA)

-- Vérifier la dimension
SELECT array_length(embedding) FROM f_offre LIMIT 1;
-- Output: 768
```

---

# Phase 4 : Processus de Recherche (Online)

## Explication Méthodologique

Lorsqu'un utilisateur lance une recherche, le système doit :
1. **Encoder** la requête utilisateur en embedding 768D
2. **Calculer** la similarité entre ce vecteur et tous les embeddings stockés
3. **Trier** par score décroissant et retourner les TOP-K résultats

La clé de l'optimisation réside dans le **compute pushdown** : déléguer le calcul au moteur SQL.

---

## Étape 1 : Encodage de la Requête

### Code Python

```python
def semantic_search(query: str, model, top_k=50):
    # Encoder la requête utilisateur
    query_embedding = model.encode(query, convert_to_numpy=True)
    # Output: array de shape (768,)
    
    # Convertir en liste Python pour DuckDB
    embedding_list = query_embedding.tolist()
    # Output: [0.123, -0.456, 0.789, ...]
```

### Application au Fil Rouge

```python
query = "Data Engineer Lyon"

query_embedding = model.encode(query)
# [0.221, -0.534, 0.801, 0.143, ..., 0.398]
#  ↑ 768 valeurs
```

**Durée** : ~50 ms (CPU moderne)

---

## Étape 2 : Calcul de Similarité (Compute Pushdown)

### Concept : Server-side vs Client-side

#### ❌ Approche Client-side (Naïve)

```python
# Récupérer TOUS les embeddings en Python
query = "SELECT job_id, embedding FROM f_offre"
results = con.execute(query).fetchdf()  # 4377 lignes × 768 colonnes

# Calculer similarité en Python
similarities = []
for idx, row in results.iterrows():
    sim = cosine_similarity(query_embedding, row['embedding'])
    similarities.append((row['job_id'], sim))

# Trier
similarities.sort(key=lambda x: x[1], reverse=True)
top_results = similarities[:50]
```

**Problèmes** :
1. **Transfert réseau** : 4377 × 3 KB = **13 MB** de données transférées (MotherDuck → Python)
2. **RAM Python** : 13 MB chargés en mémoire
3. **Latence** : Transfert + calcul Python = **1-2 secondes**
4. **Scalabilité** : Si 50 000 offres → 150 MB transférés !

---

#### ✅ Approche Server-side (Compute Pushdown)

```python
# Calcul DANS MotherDuck (SQL)
query_sql = f"""
SELECT 
    f.job_id,
    f.title,
    l.ville,
    c.type_contrat,
    -- Calcul de similarité côté SQL
    array_cosine_similarity(
        f.embedding,              -- Vecteur stocké (768D)
        ?::FLOAT[768]            -- Vecteur de la requête (paramètre)
    ) AS similarity_score
FROM f_offre f
LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
LEFT JOIN d_contrat c ON f.id_contrat = c.id_contrat
WHERE f.embedding IS NOT NULL
ORDER BY similarity_score DESC
LIMIT {top_k}
"""

results = con.execute(query_sql, [embedding_list]).fetchdf()
# Retourne SEULEMENT les 50 meilleurs résultats
```

**Avantages** :
1. **Transfert minimal** : Seulement 50 lignes × ~1 KB = **50 KB** (vs 13 MB)
2. **Calcul optimisé** : DuckDB utilise du C++ vectorisé (SIMD)
3. **Latence** : ~100-200 ms (vs 1-2 secondes)
4. **Scalabilité** : Latence quasi-linéaire en O(n)

---

### Formule Mathématique : Similarité Cosinus

Pour deux vecteurs $\mathbf{u}$ et $\mathbf{v}$ de dimension 768 :

$$
\text{cosine\_similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{||\mathbf{u}|| \times ||\mathbf{v}||} = \frac{\sum_{i=1}^{768} u_i \times v_i}{\sqrt{\sum_{i=1}^{768} u_i^2} \times \sqrt{\sum_{i=1}^{768} v_i^2}}
$$

**Propriétés** :
- **Domaine** : $[-1, 1]$
  - $+1$ : Vecteurs identiques (parfaite similarité)
  - $0$ : Vecteurs orthogonaux (aucune similarité)
  - $-1$ : Vecteurs opposés (anti-similarité)
- **Indépendant de la norme** : Seule la **direction** compte, pas la magnitude
- **Symétrique** : $\cos(\mathbf{u}, \mathbf{v}) = \cos(\mathbf{v}, \mathbf{u})$

---

### Implémentation DuckDB

DuckDB fournit la fonction native `array_cosine_similarity()` :

```sql
array_cosine_similarity(
    array1: FLOAT[N],
    array2: FLOAT[N]
) → FLOAT
```

**Exemple** :
```sql
SELECT array_cosine_similarity(
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0]
);
-- Output: 1.0 (vecteurs identiques)

SELECT array_cosine_similarity(
    [1.0, 0.0],
    [0.0, 1.0]
);
-- Output: 0.0 (vecteurs orthogonaux)
```

---

## Étape 3 : Filtres Hybrides (NLP + SQL)

Le système combine **recherche sémantique** (NLP) et **filtres SQL** (localisation, contrat) :

```python
where_clauses = ["f.embedding IS NOT NULL"]

if ville_filter == "Lyon":
    where_clauses.append("l.ville = 'Lyon'")

if contrat_filter == "CDI":
    where_clauses.append("c.type_contrat = 'CDI'")

where_sql = " AND ".join(where_clauses)
# → "f.embedding IS NOT NULL AND l.ville = 'Lyon' AND c.type_contrat = 'CDI'"
```

**Requête SQL complète** :
```sql
SELECT 
    f.job_id,
    f.title,
    l.ville,
    c.type_contrat,
    array_cosine_similarity(f.embedding, ?::FLOAT[768]) AS similarity_score
FROM f_offre f
LEFT JOIN d_localisation l ON f.id_ville = l.id_ville
LEFT JOIN d_contrat c ON f.id_contrat = c.id_contrat
WHERE 
    f.embedding IS NOT NULL 
    AND l.ville = 'Lyon'        -- Filtre SQL
    AND c.type_contrat = 'CDI'  -- Filtre SQL
ORDER BY similarity_score DESC  -- Tri NLP
LIMIT 50
```

**Architecture hybride** :
1. **Filtres SQL** : Réduisent le pool (4377 offres → 200 offres à Lyon)
2. **Similarité NLP** : Trie les 200 offres par pertinence sémantique
3. **TOP-K** : Retourne les 50 meilleures

**Avantage** : Performances optimales (~50 ms au lieu de 200 ms)

---

# Phase 5 : Interprétation des Résultats

## Application au Fil Rouge

### Requête : "Data Engineer Lyon"

```python
query = "Data Engineer Lyon"
query_embedding = model.encode(query)
# [0.221, -0.534, 0.801, 0.143, ..., 0.398]
```

---

### Calcul de Similarité pour Chaque Offre

#### Offre `job_001` : Data Engineer (Lyon)

```python
# Document enrichi (Phase 1)
doc_001 = "Data Engineer | DataCorp | CDI | Lyon | 69001 | 
           Construction de pipelines de données pour..."

# Embedding (Phase 3)
emb_001 = [0.234, -0.521, 0.789, 0.156, ..., 0.412]
```

**Calcul** :
$$
\text{sim}(query, job\_001) = \frac{query \cdot emb\_001}{||query|| \times ||emb\_001||}
$$

En supposant que les embeddings sont **normalisés** (norme = 1), le calcul se simplifie :
$$
\text{sim} = query \cdot emb\_001 = \sum_{i=1}^{768} q_i \times e_i
$$

```python
# Produit scalaire (simplifié pour illustration)
similarity = (
    0.221 × 0.234 +    # Dimension 1
    -0.534 × -0.521 +  # Dimension 2
    0.801 × 0.789 +    # Dimension 3
    ...                # 765 autres dimensions
    0.398 × 0.412      # Dimension 768
)
# ≈ 0.87 (score élevé)
```

**Interprétation** : Score de **0.87 (87%)** indique une **très haute similarité**.

---

#### Offre `job_002` : Ingénieur Données (Lyon)

```python
doc_002 = "Ingénieur Données | TechCorp | CDI | Lyon | 69002 | 
           Développement d'architectures Big Data..."

emb_002 = [0.218, -0.509, 0.776, 0.141, ..., 0.389]
```

**Calcul** :
$$
\text{sim}(query, job\_002) \approx 0.82
$$

**Interprétation** : Score de **0.82 (82%)** légèrement inférieur à `job_001` car :
- Terme "Ingénieur Données" (français) vs "Data Engineer" (anglais)
- Le modèle comprend la **synonymie** mais privilégie la correspondance exacte

---

#### Offre `job_003` : Data Analyst (Paris)

```python
doc_003 = "Data Analyst | BizCorp | CDI | Paris | 75001 | 
           Analyse de données business pour..."

emb_003 = [0.198, -0.487, 0.712, 0.133, ..., 0.351]
```

**Calcul** :
$$
\text{sim}(query, job\_003) \approx 0.65
$$

**Interprétation** : Score de **0.65 (65%)** modéré car :
- **Rôle proche** : "Data Analyst" vs "Data Engineer" (domaine Data)
- **Localisation différente** : "Paris" vs "Lyon" (pénalité sémantique)

---

#### Offre `job_004` : Boulanger (Lyon)

```python
doc_004 = "Boulanger | Boulangerie Artisanale | CDI | Lyon | 69001 | 
           Fabrication de pain et viennoiseries..."

emb_004 = [-0.023, 0.156, -0.089, 0.021, ..., -0.102]
```

**Calcul** :
$$
\text{sim}(query, job\_004) \approx 0.12
$$

**Interprétation** : Score de **0.12 (12%)** très faible car :
- **Aucun lien sémantique** : "Boulanger" vs "Data Engineer"
- **Vocabulaire disjoint** : "pain", "pétrissage" vs "données", "pipelines"
- Seule correspondance : **"Lyon"** (insuffisant)

---

### Classement Final (TOP 50)

```sql
SELECT 
    job_id, 
    title, 
    ville,
    ROUND(similarity_score * 100, 1) AS score_pct
FROM results
ORDER BY similarity_score DESC
LIMIT 4;
```

**Résultat** :
| Rang | job_id | title | ville | score_pct |
|------|--------|-------|-------|-----------|
| 1 | job_001 | Data Engineer | Lyon | **87.0%** |
| 2 | job_002 | Ingénieur Données | Lyon | **82.0%** |
| 3 | job_003 | Data Analyst | Paris | **65.0%** |
| ... | ... | ... | ... | ... |
| 2341 | job_004 | Boulanger | Lyon | **12.0%** |

**Analyse** :
- `job_001` et `job_002` : **TOP 2** (même ville, même domaine)
- `job_003` : Rang moyen (domaine proche, ville différente)
- `job_004` : **Très bas** (aucun lien sémantique)

---

## Visualisation Géométrique (Projection 2D)

```
              Score de Similarité
                    │
                1.0 │  ● job_001 (Data Engineer Lyon)
                    │  ● job_002 (Ingénieur Données Lyon)
                    │
                0.8 │
                    │
                0.6 │          ● job_003 (Data Analyst Paris)
                    │
                0.4 │
                    │
                0.2 │
                    │                             ● job_004 (Boulanger Lyon)
                0.0 │
                    └─────────────────────────────────────────→
                       Distance Sémantique à la Requête
```

---

### Optimisations Futures

1. **Index HNSW (Hierarchical Navigable Small World)** :
   - Recherche approximative en O(log n) au lieu de O(n)
   - DuckDB ne supporte pas nativement (alternative : FAISS, Pinecone)

2. **Quantization** :
   - Réduire 768D → 384D ou 256D
   - Trade-off : -20% précision, +50% vitesse

3. **Caching** :
   - Mettre en cache les requêtes fréquentes ("Data Scientist Paris")
   - Redis ou Memcached côté serveur

---

# Références Bibliographiques

## Embeddings et Sentence Transformers

1. **Reimers, N., & Gurevych, I. (2019)**  
   *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"*  
   EMNLP 2019.  

2. **Reimers, N., & Gurevych, I. (2020)**  
   *"Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation"*  
   EMNLP 2020.  

## Similarité Cosinus et Recherche Vectorielle

3. **Salton, G., & McGill, M. J. (1983)**  
   *Introduction to Modern Information Retrieval*  
   McGraw-Hill.  

4. **Johnson, J., Douze, M., & Jégou, H. (2019)**  
   *"Billion-scale similarity search with GPUs"*  
   IEEE Transactions on Big Data.  

## Star Schema et Data Warehousing

5. **Kimball, R., & Ross, M. (2013)**  
   *The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling (3rd ed.)*  
   Wiley.  

## DuckDB et Compute Pushdown

6. **Raasveldt, M., & Mühleisen, H. (2019)**  
   *"DuckDB: an Embeddable Analytical Database"*  
   SIGMOD 2019.  
---

**Auteurs** : Équipe RUCHE  
**Date** : Janvier 2026  
**Version** : 1.0
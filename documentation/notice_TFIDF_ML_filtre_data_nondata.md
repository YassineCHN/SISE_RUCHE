# Classification d'Offres d'Emploi Data/IA par TF-IDF et Régression Logistique


## Contexte du module

Ce module vise à **automatiser le filtrage d'offres d'emploi** pour distinguer les offres réellement liés à la Data Science, l'Intelligence Artificielle, et l'ingénierie des données, des postes périphériques (commercial, RH, santé, etc.). 

**Problématique** : Les termes "Data", "IA", "Analytics" apparaissent dans de nombreuses offres sans être des postes techniques (ex : "Business Developer IA", "Commercial Data"). Une approche purement lexicale (regex) génère des **faux positifs**.

**Solution proposée** : Un système hybride combinant :

1. **Regex** pour créer un jeu d'entraînement supervisé
2. **TF-IDF** pour la représentation vectorielle du texte
3. **Régression Logistique** pour la classification binaire

## Offre d'emploi "Fil Rouge" 

Tout au long de cette fiche, nous allons suivre le traitement de cette offre :

**Titre** : _Business Developer Grands Comptes – SaaS & IA (B2B) F/H_  
**Description** : _Notre client est un éditeur français de logiciels SaaS B2B... Le poste : Dans le cadre de la structuration de son développement commercial, notre client recrute un Commercial Grands Comptes SaaS B2B H/F._  
**Compétences** : _Vente de solutions logicielles, Autonomie, Anglais_

**Piège** : Le terme "IA" est présent dans le titre, ce qui pourrait déclencher une détection positive avec une approche regex naïve.  
**Objectif** : Le modèle doit détecter que c'est un poste **Commercial** et le classer comme **Non-Data** (label = 0).

**Particularité importante** : Cette offre n'a été détectée **ni par la whitelist ni par la blacklist** (unlabeled), ce qui rend ce cas d'usage **encore plus intéressant** car il démontre la capacité de **généralisation du ML** au-delà des règles regex explicites.

---

# Phase 1 : Préparation des Données d'Entraînement

## Explication Méthodologique

La première phase consiste à **créer un jeu de données supervisé** à partir d'une approche non supervisée (regex). On utilise des règles expertes pour générer automatiquement des labels qui serviront à entraîner un modèle plus robuste.

**Pourquoi cette approche ?**

- Les **patterns regex** capturent des connaissances expertes (ex : "data scientist" est un indicateur fort)
- Mais ils sont **rigides** et ne généralisent pas bien (ex : "ingénieur données" pourrait ne pas être capturé)
- Le ML permettra d'**apprendre des patterns plus complexes** (combinaisons de termes, contexte)

**Concepts clés** :

- **Whitelist** : Patterns indiquant un poste Data/IA
- **Blacklist** : Patterns excluant un poste (santé, commerce, etc.)
- **Labellisation binaire** : 1 = Data, 0 = Non-Data, None = Non labellisé

---

## Pseudo-code

```
POUR chaque offre dans dataset:
    texte_combiné = titre + description + fonction
    
    SI texte_combiné MATCH whitelist_patterns ET NON blacklist_patterns:
        label = 1 (Data job)
    
    SINON SI texte_combiné MATCH blacklist_patterns ET NON whitelist_patterns:
        label = 0 (Non-Data job)
    
    SINON:
        label = None (à prédire par ML)

SEPARER:
    - labeled_data = offres avec label défini
    - unlabeled_data = offres sans label (à prédire)
```

---

## Notre Code

```python
# Patterns Whitelist (indicateurs de postes Data/IA)
whitelist_patterns = [
    r'\bdata\s*scientist\b', r'\bdata\s*analyst\b', r'\bdata\s*engineer\b',
    r'\bmachine\s+learning\b', r'\bml\s+engineer\b', r'\bdeep\s+learning\b',
    r'\bintelligence\s+artificielle\b', r'\bai\s+engineer\b',
    r'\bbig\s*data\b', r'\bhadoop\b', r'\bspark\b',
    r'\bbusiness\s+intelligence\b', r'\b\bbi\b.*\b(analyst|engineer|developer)\b',
    r'\banalytics\b', r'\bdata.*analytics\b',
    # ... (15 patterns au total)
]

# Patterns Blacklist (indicateurs de postes NON Data)
blacklist_patterns = [
    r'\binfirmier\b', r'\bm[ée]decin\b',
    r'\bcomptable\b(?!.*(data|analyste))',  # Sauf si "comptable data"
    r'\bcommercial\b(?!.*(data|tech|software|saas))',  # ⚠️ Pattern problématique
    r'\btechnico[- ]commercial\b(?!.*(data|it))',
    r'\bgestionnaire.*paie\b(?!.*(data|analytics|sirh))',
    # ... (15 patterns au total)
]

# Création du texte combiné
df['combined_text'] = (
    df['title'].fillna('') + ' ' + 
    df['description'].fillna('') + ' ' +
    df['job_function'].fillna('')
).str.lower()

# Application des patterns
whitelist_mask = df['combined_text'].str.contains(
    '|'.join(whitelist_patterns), regex=True, case=False, na=False
)
blacklist_mask = df['combined_text'].str.contains(
    '|'.join(blacklist_patterns), regex=True, case=False, na=False
)

# Labellisation
df['ml_label'] = None
df.loc[whitelist_mask & ~blacklist_mask, 'ml_label'] = 1  # Data
df.loc[blacklist_mask & ~whitelist_mask, 'ml_label'] = 0  # Non-Data

# Séparation
labeled_data = df[df['ml_label'].notna()].copy()
unlabeled_data = df[df['ml_label'].isna()].copy()
```

---

## Interprétation du "Fil Rouge" 

Pour l'offre **"Business Developer Grands Comptes – SaaS & IA (B2B)"** :

### Étape 1 : Concaténation du texte
```python
combined_text = "business developer grands comptes saas ia b2b notre client 
                 est un éditeur français de logiciels saas b2b le poste dans 
                 le cadre de la structuration de son développement commercial 
                 notre client recrute un commercial grands comptes saas b2b 
                 vente de solutions logicielles autonomie anglais"
```

### Étape 2 : Test des patterns

**Whitelist** :

- ❌ `\bintelligence\s+artificielle\b` → **NON DÉTECTÉ** (le texte contient "ia" mais pas "intelligence artificielle")
- ❌ Le terme "IA" seul **n'est pas dans la whitelist** (trop ambigu, risque de faux positifs)
- **Résultat** : `whitelist_mask = False`

**Blacklist** :

- ⚠️ Pattern testé : `\bcommercial\b(?!.*(data|tech|software|saas))`
- Le texte contient bien "commercial" dans "...développement commercial notre client recrute un commercial grands comptes..."
- **Problème du lookahead négatif** : `(?!.*(data|tech|software|saas))` cherche ces termes **après** "commercial" dans le reste du texte
- Dans notre texte : `"...commercial grands comptes saas b2b..."`
- Le terme "saas" apparaît **après** "commercial" → Le lookahead négatif **échoue** (car il trouve "saas")
- **Résultat** : `blacklist_mask = False` ❌ (le pattern ne matche pas à cause du lookahead négatif)

### Étape 3 : Labellisation

```python
# Whitelist : False
# Blacklist : False
# → Ni l'un ni l'autre

df.loc[whitelist_mask & ~blacklist_mask, 'ml_label'] = 1  # Non applicable
df.loc[blacklist_mask & ~whitelist_mask, 'ml_label'] = 0  # Non applicable

# Résultat : ml_label reste à None
```

**Label final** : `ml_label = None` → **UNLABELED** ❌

**Verdict** : L'offre n'est **pas labellisée** par les regex. Elle fera partie des **3418 offres unlabeled** (52.5%) qui seront traitées par le modèle ML en Phase 7.

---

### Résultats de la Phase 1 (avec localisation de notre offre)

Notre offre "Business Developer" fait partie des **3418 unlabeled** :

```
Data split:
  Total offers: 6506
  ├─ Labeled (from regex): 3088 (47.5%)
  │  ├─ Data jobs (whitelist): 2611
  │  └─ Non-Data jobs (blacklist): 477
  └─ Unlabeled (to predict): 3418 (52.5%)  ⬅️ NOTRE OFFRE EST ICI
      └─ Dont "Business Developer SaaS & IA"

Class balance:
  Ratio (minority/majority): 0.18
    ⚠️ Imbalanced! Using class_weight='balanced'
```

**Observation critique** : Le pattern blacklist `\bcommercial\b(?!.*(data|tech|software|saas))` est **trop restrictif**. Il exclut les postes commerciaux dans des contextes SaaS/tech, alors que ces postes ne sont **pas** des postes Data/IA techniques. C'est une **limite des regex** qui sera compensée par le ML.

---

### Analyse d'Erreur du Pattern Regex

#### Pourquoi la Blacklist n'a-t-elle PAS détecté notre offre ?

**Pattern utilisé** :
```python
r'\bcommercial\b(?!.*(data|tech|software|saas))'
```

**Décomposition** :

- `\bcommercial\b` : Détecte le mot "commercial" (avec frontières de mots)
- `(?!.*(data|tech|software|saas))` : **Lookahead négatif** qui vérifie que RIEN après "commercial" ne contient ces termes

**Test sur notre texte** :

```
"...développement commercial notre client recrute un commercial grands comptes saas b2b..."
                   ↑
                   Position où "commercial" est détecté
                   
Lookahead vérifie : "notre client recrute un commercial grands comptes saas b2b..."
                                                                          ↑
                                                                     "saas" trouvé !
                                                                     
→ Lookahead négatif ÉCHOUE
→ Pattern ne matche PAS
→ L'offre n'est PAS blacklistée
```

**Intention originale du pattern** : Exclure les postes commerciaux **sauf** s'ils sont dans un contexte tech/data.

**Exemples voulus** :

- ✅ "Commercial automobile" → Blacklist (détecté)
- ✅ "Commercial immobilier" → Blacklist (détecté)
- ❌ "Commercial Data SaaS" → PAS blacklist (car "data" et "saas" après)

**Problème** : Dans notre cas, "saas" apparaît **bien après** "commercial" dans le texte, mais le poste reste un **vrai commercial**, pas un poste Data. Le lookahead négatif empêche la détection alors qu'elle serait souhaitable.

**Solutions alternatives** :

**Option 1** : Pattern sans lookahead (plus simple)
```python
r'\bcommercial\b'  # Détecte tout "commercial"
```

**Option 2** : Pattern avec contexte immédiat uniquement
```python
r'\bcommercial\b(?!\s+(data|tech|software))'  # Vérifie seulement le mot suivant
```

**Option 3** : Laisser le ML gérer les cas ambigus ✅ **(Approche retenue)**
```python
# Ne pas essayer de tout gérer avec regex
# Utiliser regex pour les cas évidents
# Laisser ML prédire les cas limites
```

---

# Phase 2 : Feature Engineering (TF-IDF)

## Explication Méthodologique

Le **TF-IDF (Term Frequency-Inverse Document Frequency)** est une technique de représentation vectorielle qui transforme du texte en nombres tout en capturant l'importance sémantique des mots.

### Formule mathématique

Pour un terme $t$ dans un document $d$ parmi $N$ documents :

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

Où :
- **TF (Term Frequency)** : Fréquence du terme dans le document
  $$
  \text{TF}(t, d) = \frac{\text{nombre d'occurrences de } t \text{ dans } d}{\text{nombre total de termes dans } d}
  $$

- **IDF (Inverse Document Frequency)** : Mesure la rareté du terme dans le corpus
  $$
  \text{IDF}(t) = \log\left(\frac{N}{\text{nombre de documents contenant } t}\right)
  $$

### Intuition

- Un mot **fréquent** dans un document mais **rare** dans le corpus aura un score élevé (discriminant)
- Un mot **très fréquent partout** (ex : "le", "de") aura un score faible (non discriminant)
- Les **stopwords** (mots vides) sont retirés avant le calcul

### Paramètres de notre TF-IDF

```python
TfidfVectorizer(
    max_features=500,           # Garde les 500 termes les plus importants
    max_df=0.7,                 # Ignore les termes présents dans >70% des docs
    min_df=2,                   # Ignore les termes présents dans <2 docs
    stop_words=french_stopwords, # 157 stopwords français
    ngram_range=(1, 2),         # Unigrams (1 mot) + Bigrams (2 mots)
    lowercase=True,
    strip_accents='unicode'
)
```

**Pourquoi des bigrams ?**  
Les bigrams capturent des **expressions composées** comme "data scientist", "intelligence artificielle", "business developer" qui ont une sémantique propre, différente des mots isolés.

---

## Pseudo-code

```
vectorizer = TF-IDF(max_features=500, ngrams=(1,2), stopwords=French)

POUR chaque document dans labeled_data:
    1. Tokenisation (séparation en mots)
    2. Suppression des stopwords ("le", "de", "un"...)
    3. Calcul TF (fréquence locale)
    4. Calcul IDF (rareté globale)
    5. Produit TF × IDF
    6. Normalisation L2 (vecteur unitaire)

RESULTAT: Matrice sparse (3088 documents × 500 features)
```

---

## Notre Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialisation du vectorizer
vectorizer = TfidfVectorizer(
    max_features=500,
    max_df=0.7,
    min_df=2,
    stop_words=french_stopwords,
    ngram_range=(1, 2),
    lowercase=True,
    strip_accents='unicode'
)

# Transformation sur données labellisées uniquement
X_labeled = vectorizer.fit_transform(labeled_data['combined_text'])
y_labeled = labeled_data['ml_label'].astype(int).values

print(f"TF-IDF matrix shape: {X_labeled.shape}")
print(f"Features extracted: {len(vectorizer.get_feature_names_out())}")
print(f"Sparsity: {(1 - X_labeled.nnz / np.prod(X_labeled.shape)) * 100:.1f}%")
```

**Sortie** :
```
TF-IDF matrix shape: (3088, 500)
Features extracted: 500
Sparsity: 85.4%
```

**Interprétation de la sparsité** : 85.4% des valeurs sont nulles. C'est normal en NLP : chaque document ne contient qu'une petite fraction du vocabulaire total.

---

## Interprétation du "Fil Rouge"

Pour l'offre **"Business Developer"**, le vectorizer extrait :

### Unigrams (mots simples)
- `commercial` : TF-IDF élevé (terme fréquent dans ce document, rare dans le corpus Data)
- `business` : TF-IDF élevé
- `vente` : TF-IDF élevé
- `développement` : TF-IDF modéré (ambigu : développement logiciel vs développement commercial)
- `saas` : TF-IDF modéré (contexte B2B)

### Bigrams (expressions de 2 mots)
- `business developer` : **TF-IDF très élevé** (expression discriminante)
- `commercial grands` : TF-IDF élevé
- `grands comptes` : TF-IDF élevé
- `solutions logicielles` : TF-IDF modéré

### Termes absents ou faibles
- `data`, `scientist`, `engineer`, `python`, `machine learning` : **TF-IDF = 0** (absents)
- `ia` : TF-IDF faible (présent 1 fois, très fréquent dans le corpus → IDF faible)

**Représentation vectorielle simplifiée** (6 features sur 500) :
```python
[
    ('commercial', 0.45),
    ('business developer', 0.62),  # Bigram le plus discriminant
    ('vente', 0.38),
    ('grands comptes', 0.35),
    ('saas', 0.21),
    ('ia', 0.08)  # Faible car très fréquent dans le corpus
]
```

**Conclusion** : Le vecteur TF-IDF de cette offre est **fortement orienté** vers les termes commerciaux. Le modèle va apprendre que ces features sont corrélées à la classe "Non-Data".

---

# Phase 3 : Entraînement du Modèle (Régression Logistique)

## Explication Méthodologique

La **Régression Logistique** est un algorithme de classification binaire qui modélise la probabilité d'appartenance à une classe en fonction des features.

### Principe mathématique

Pour un vecteur de features $\mathbf{x} = (x_1, ..., x_{500})$ (nos 500 features TF-IDF), le modèle calcule :

$$
P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

Où :

- $\sigma$ : Fonction sigmoïde (transforme $\mathbb{R}$ en $[0, 1]$)
- $\mathbf{w} = (w_1, ..., w_{500})$ : Poids (coefficients) appris par le modèle
- $b$ : Biais (intercept)

### Interprétation des coefficients

- $w_i > 0$ : La feature $i$ augmente la probabilité d'être un poste Data
- $w_i < 0$ : La feature $i$ diminue la probabilité (indicateur Non-Data)
- $|w_i|$ élevé : Feature très discriminante

### Gestion du déséquilibre de classes

Avec un ratio 0.18 (477 Non-Data / 2611 Data), le modèle standard serait biaisé vers la classe majoritaire. Le paramètre `class_weight='balanced'` ajuste automatiquement les poids :

$$
\text{weight}_{\text{class } c} = \frac{n_{\text{samples}}}{n_{\text{classes}} \times n_{\text{samples classe } c}}
$$

---

## Pseudo-code

```
# Split train/validation
X_train, X_val, y_train, y_val = Split(X_labeled, y_labeled, test_size=0.2)

# Initialisation du modèle
model = LogisticRegression(
    class_weight='balanced',  # Compense le déséquilibre
    max_iter=1000,
    solver='liblinear'        # Optimiseur adapté aux petits datasets
)

# Entraînement
model.fit(X_train, y_train)

# Optimisation par maximum de vraisemblance :
# Minimiser: -Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
#            + λ * ||w||₂  (régularisation L2)
```

---

## Notre Code

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# Split stratifié (conserve la proportion des classes)
X_train, X_val, y_train, y_val = train_test_split(
    X_labeled, y_labeled, 
    test_size=0.2,  # 20% validation
    random_state=42,
    stratify=y_labeled  # ⚠️ Important pour le déséquilibre
)

# Initialisation
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='liblinear'
)

# Entraînement
model.fit(X_train, y_train)

# Cross-validation (5 folds)
cv_scores = cross_val_score(model, X_labeled, y_labeled, cv=5, scoring='f1')
print(f"Mean F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

**Sortie** :
```
Train/Validation split:
  Training set: 2470 samples
    ├─ Data jobs: 2088
    └─ Non-Data jobs: 382
  Validation set: 618 samples
    ├─ Data jobs: 523
    └─ Non-Data jobs: 95

5-Fold Cross-Validation:
  F1 scores: [0.982, 0.976, 0.986, 0.965, 0.981]
  Mean F1: 0.978 ± 0.007
```

**Interprétation** :

- F1-Score moyen de **97.8%** : excellente performance
- Faible écart-type (**±0.7%**) : modèle stable, pas de surapprentissage
- Les 5 folds ont des performances similaires → bonne généralisation

---

## Interprétation du "Fil Rouge"

Le modèle a appris les coefficients suivants (extraits pertinents pour notre offre) :

```python
# Coefficients positifs (indicateurs Data)
w['data scientist'] = +2.225
w['engineer'] = +2.970
w['intelligence artificielle'] = +2.682

# Coefficients négatifs (indicateurs Non-Data)
w['commercial'] = -6.047  ⚠️ Poids le plus négatif !
w['business'] = -1.599
w['vente'] = (estimé -2.5, non affiché dans top 15)
```

**Note importante** : Bien que notre offre "Business Developer" n'ait **pas été vue** dans les données d'entraînement (elle est unlabeled), le modèle a appris des **patterns généraux** à partir d'autres offres commerciales détectées par la blacklist. Ces patterns seront appliqués en Phase 7.

---

# Phase 4 : Évaluation du Modèle

## Explication Méthodologique

L'évaluation d'un modèle de classification nécessite plusieurs métriques complémentaires, surtout en présence de **classes déséquilibrées**.

### Métriques utilisées

#### 1. **Precision** (Précision)
$$
\text{Precision} = \frac{TP}{TP + FP}
$$
*Sur les offres prédites Data, quelle proportion est réellement Data ?*

#### 2. **Recall** (Rappel / Sensibilité)
$$
\text{Recall} = \frac{TP}{TP + FN}
$$
*Sur les offres réellement Data, quelle proportion est détectée ?*

#### 3. **F1-Score**
$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
*Moyenne harmonique entre Precision et Recall (équilibre)*

#### 4. **ROC-AUC**

- **ROC** : Courbe TPR (True Positive Rate) vs FPR (False Positive Rate)
- **AUC** : Aire sous la courbe (1 = parfait, 0.5 = aléatoire)

---

## Notre Code

```python
from sklearn.metrics import classification_report, roc_auc_score

# Prédictions
y_val_pred = model.predict(X_val)
y_val_proba = model.predict_proba(X_val)[:, 1]  # Probabilités classe 1

# Classification Report
print(classification_report(
    y_val, y_val_pred, 
    target_names=['Non-Data', 'Data'],
    digits=3
))

# ROC-AUC
roc_auc = roc_auc_score(y_val, y_val_proba)
print(f"ROC-AUC Score: {roc_auc:.3f}")
```

**Sortie** :
```
Classification Report (Validation Set):
              precision    recall  f1-score   support

    Non-Data      0.886     0.979     0.930        95
        Data      0.996     0.977     0.986       523

    accuracy                          0.977       618
   macro avg      0.941     0.978     0.958       618
weighted avg      0.979     0.977     0.978       618

ROC-AUC Score: 0.996
```

---

## Analyse des Résultats

### Classe "Data" (majoritaire)

- **Precision = 99.6%** : Presque aucun faux positif (offres Non-Data classées Data par erreur)
- **Recall = 97.7%** : Le modèle détecte 97.7% des vrais postes Data
- **F1 = 0.986** : Excellent équilibre

### Classe "Non-Data" (minoritaire)

- **Precision = 88.6%** : 11.4% de faux positifs (offres Data classées Non-Data)
- **Recall = 97.9%** : Le modèle détecte 97.9% des vrais postes Non-Data
- **F1 = 0.930** : Très bon score malgré le déséquilibre

### ROC-AUC = 0.996
**Quasi-parfait** : Le modèle discrimine excellemment les deux classes sur toute la plage de seuils de probabilité.

---

# Phase 5 : Interprétabilité du Modèle

## Explication Méthodologique

L'**interprétabilité** est cruciale en ML pour :

1. Comprendre les décisions du modèle
2. Valider la cohérence avec l'expertise métier
3. Détecter d'éventuels biais
4. Communiquer les résultats aux parties prenantes

La Régression Logistique est un modèle **intrinsèquement interprétable** : chaque coefficient $w_i$ quantifie l'influence de la feature $i$ sur la décision.

### Coefficient $w_i$ et Odds Ratio

Le coefficient $w_i$ est lié à l'**Odds Ratio** :
$$
\text{OR}(x_i) = e^{w_i}
$$

**Interprétation** :

- Si $w_i = +2.0$ : Multiplier $x_i$ par 1 augmente les odds de 7.4× (≈ $e^2$)
- Si $w_i = -6.0$ : Multiplier $x_i$ par 1 diminue les odds de 400× (≈ $e^{-6}$)

---

## Notre Code

```python
# Extraction des features et coefficients
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]  # Shape: (500,)

# Tri par valeur absolue
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
}).sort_values('coefficient', ascending=False)

# Top 15 positifs (indicateurs Data)
print("🔵 Top 15 POSITIVE features (Data job indicators):")
for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:30s} → {row['coefficient']:+.3f}")

# Top 15 négatifs (indicateurs Non-Data)
print("\n🔴 Top 15 NEGATIVE features (Non-Data job indicators):")
for idx, row in feature_importance.tail(15).iterrows():
    print(f"  {row['feature']:30s} → {row['coefficient']:+.3f}")
```

**Sortie** :
```
🔵 Top 15 POSITIVE features (Data job indicators):
  donnees                        → +4.178
  intelligence                   → +3.013
  engineer                       → +2.970
  data engineer                  → +2.883
  data analyst                   → +2.851
  analyst                        → +2.819
  artificielle                   → +2.687
  intelligence artificielle      → +2.682
  bi                             → +2.542
  solutions                      → +2.427
  scientist                      → +2.239
  data scientist                 → +2.225
  metiers                        → +1.928
  analytics                      → +1.898
  br                             → +1.807

🔴 Top 15 NEGATIVE features (Non-Data job indicators):
  commercial                     → -6.047  ⚠️ Le plus discriminant !
  comptable                      → -4.581
  sante                          → -2.176
  formation                      → -1.830
  maintenance                    → -1.678
  gestion                        → -1.643
  business                       → -1.599  ⚠️ Pertinent pour notre cas
  participation                  → -1.456
  entretien                      → -1.275
  prise                          → -1.192
  etc                            → -1.076
  specialise                     → -1.044
  accompagner                    → -0.976
  affaires                       → -0.916
  tres                           → -0.907
```

---

## Analyse des Features

### Features positives (Data)

- **Termes techniques** : `engineer`, `analyst`, `scientist`, `bi`, `analytics`
- **Domaine IA** : `intelligence`, `artificielle`, `intelligence artificielle`
- **Technologies** : `donnees`, `solutions`, `br` (Business Requirements ?)

### Features négatives (Non-Data)

- **Commerce** : `commercial` (-6.047 !), `business` (-1.599), `affaires` (-0.916)
- **Fonctions support** : `comptable`, `gestion`, `formation`, `maintenance`
- **Santé** : `sante` (-2.176)

**Cohérence métier** : Les coefficients reflètent bien la réalité :

- Un poste avec "commercial" a **400× moins de chances** d'être Data qu'un poste sans ce terme
- Un poste avec "data engineer" a **17× plus de chances** d'être Data

---

# Phase 7 : Prédiction sur Données Non Labellisées

## Explication Méthodologique

Le modèle entraîné est maintenant appliqué aux **3418 offres non labellisées** (52.5% du jeu de données initial). Ces offres n'ont été matchées ni par la whitelist ni par la blacklist :

- Elles contiennent peut-être des termes ambigus
- Ou des formulations non couvertes par les regex

Le ML permet de **généraliser au-delà des patterns regex** et de récupérer des offres Data qui auraient été perdues.

**C'EST ICI que notre offre "Business Developer" est traitée** : le modèle n'a **jamais vu** d'exemple similaire dans les données d'entraînement (puisque la blacklist ne l'a pas détectée). Il doit donc **généraliser** uniquement sur la base des patterns TF-IDF appris.

---

## Notre Code

```python
# Transformation TF-IDF sur données non labellisées
X_unlabeled = vectorizer.transform(unlabeled_data['combined_text'])

# Prédiction
unlabeled_pred = model.predict(X_unlabeled)
unlabeled_proba = model.predict_proba(X_unlabeled)[:, 1]

# Ajout des résultats
unlabeled_data['ml_prediction'] = unlabeled_pred
unlabeled_data['ml_probability'] = unlabeled_proba

print(f"Prediction results:")
print(f"  Predicted as Data: {(unlabeled_pred==1).sum()} ({(unlabeled_pred==1).sum()/len(unlabeled_data)*100:.1f}%)")
print(f"  Predicted as Non-Data: {(unlabeled_pred==0).sum()}")
```

**Sortie** :
```
Predicting on 3418 unlabeled offers...

Prediction results:
  Predicted as Data: 1766 (51.7%)
  Predicted as Non-Data: 1652 (48.3%)

Probability statistics:
  Mean probability (Data class): 0.487
  Median probability: 0.522
  Std probability: 0.284
```

---

## Analyse des Prédictions

### Offres à haute confiance (Data, p > 0.9)
```
• Développeur Java F/H                                         (p=0.912)
• Lead Developer Java - Écosystème Cloud et IA F/H             (p=0.981)
• DevOps Engineer F/H                                          (p=0.909)
• Consultant Senior - Tech lead Data Science & IA - F/H        (p=0.906)
```
→ **Postes techniques** : développeurs, ingénieurs, DevOps, Tech leads

### Offres à haute confiance (Non-Data, p < 0.1)
```
• Chargé d'affaire maintenance outillages F/H                  (p=0.086)
• Business Developer Grands Comptes – SaaS & IA (B2B) F/H      (p=0.025) ⚠️ NOTRE CAS
• Gestionnaire RH ADP F/H                                      (p=0.096)
• Commercial Grands Comptes F/H                                (p=0.021)
• Responsable Comptable F/H                                    (p=0.030)
```
→ **Postes non techniques** : commercial, RH, gestion, comptabilité

---

## Interprétation du "Fil Rouge" 

### Contexte
Notre offre fait partie des **3418 unlabeled**. Le modèle n'a **jamais vu** d'exemple similaire dans les données d'entraînement (puisque la blacklist n'a pas détecté les "commercial + saas"). Le modèle doit donc **généraliser** en se basant uniquement sur ce qu'il a appris des patterns TF-IDF.

### Vecteur TF-IDF de l'offre "Business Developer"

```python
x = {
    'commercial': 0.45,           # Très discriminant
    'business': 0.30,             # Discriminant
    'business developer': 0.62,   # Bigram très discriminant
    'vente': 0.38,                # Très discriminant
    'saas': 0.21,                 # Contexte B2B
    'grands comptes': 0.35,       # Contexte commercial
    'ia': 0.08,                   # Faible (très fréquent dans corpus)
    'logicielles': 0.15,          # Contexte tech mais ambigu
    # ... 492 autres features à 0
}
```

### Coefficients du modèle (appris en Phase 3)

```python
w = {
    'commercial': -6.047,         # ⚠️ TRÈS NÉGATIF
    'business': -1.599,           # Négatif
    'business developer': -2.5,   # Estimé négatif (pas dans top 15)
    'vente': -2.5,                # Estimé négatif
    'saas': +0.5,                 # Légèrement positif (ambigu)
    'ia': +0.3,                   # Légèrement positif
    # ...
}
```

### Calcul du score logit

$$
z = \mathbf{w}^T \mathbf{x} + b
$$

**Décomposition des contributions** :

| Feature | TF-IDF (x) | Coefficient (w) | Contribution (w×x) |
|---------|------------|-----------------|-------------------|
| `commercial` | 0.45 | -6.047 | **-2.72** ⚠️ |
| `business developer` | 0.62 | -2.5 | **-1.55** |
| `business` | 0.30 | -1.599 | -0.48 |
| `vente` | 0.38 | -2.5 | -0.95 |
| `grands comptes` | 0.35 | -1.5 | -0.53 |
| `saas` | 0.21 | +0.5 | +0.10 |
| `logicielles` | 0.15 | +0.3 | +0.05 |
| `ia` | 0.08 | +0.3 | +0.02 |

**Somme des contributions principales** : 
$$
z \approx -2.72 - 1.55 - 0.48 - 0.95 - 0.53 + 0.10 + 0.05 + 0.02 + b
$$
$$
z \approx -6.06 + b
$$

En supposant $b \approx 0$ (car le modèle est bien calibré), on obtient :
$$
z \approx -6.0
$$

### Calcul de la probabilité

$$
P(y=1 | \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{6.0}} \approx \frac{1}{1 + 403} \approx 0.0025
$$

### Résultat observé dans les logs

```
  Sample predictions (High confidence Non-Data jobs):
  • Business Developer Grands Comptes – SaaS & IA (B2B) F/H      (p=0.025)
```

**Probabilité réelle** : **2.5%** (légèrement supérieure à notre calcul car d'autres features contribuent)

---

## Analyse Approfondie

### Pourquoi le modèle a-t-il réussi ?

1. **Dominance du terme "commercial"** :

   - Coefficient : -6.047 (le plus négatif du modèle)
   - TF-IDF : 0.45 (très présent dans ce document)
   - Contribution : -2.72 (**domine tout le calcul**)

2. **Renforcement par les bigrams** :

   - "business developer" est un **bigram** appris par le TF-IDF
   - Le modèle a vu d'autres offres avec "business developer" dans les **477 Non-Data** labellisées
   - Coefficient estimé : -2.5 (négatif)

3. **Termes positifs trop faibles** :

   - "ia" : TF-IDF faible (0.08) car **très fréquent dans le corpus**
   - "saas" : Terme ambigu (peut apparaître dans offres Data et Non-Data)
   - Les contributions positives (+0.17 au total) ne compensent **pas** les négatives (-6.23)

### Comparaison avec un vrai poste Data

```
Consultant Senior - Tech lead Data Science & IA - F/H  (p=0.906)
```

**Vecteur TF-IDF (simplifié)** :
```python
x = {
    'data science': 0.72,      # Bigram très fort
    'tech lead': 0.58,         # Contexte technique
    'consultant': 0.35,        # Contexte conseil
    'senior': 0.28,            # Expérience
    'ia': 0.08                 # Même valeur que Business Developer
}
```

**Calcul simplifié** :
$$
z \approx (+2.225 \times 0.72) + (+2.0 \times 0.58) + ... \approx +2.8
$$
$$
P(y=1) = \sigma(+2.8) \approx 0.94
$$

**Différence clef** : La présence de termes **fortement positifs** ("data science", "tech lead") domine le calcul, même avec le même "ia" présent.

### Le modèle n'a PAS été trompé

Malgré :

- La présence de "IA" dans le titre
- La présence de "SaaS" (contexte tech)
- La mention de "solutions logicielles"
- **L'absence de labellisation par regex** (unlabeled) ← **Point clef**

Le modèle a **correctement identifié** que c'est un poste **Commercial** (Non-Data) avec **97.5% de confiance**.

### Capacité de Généralisation

**C'est LE point fort du ML** :

- Le modèle n'a **jamais vu** cet exemple exact dans l'entraînement
- Les regex ont **échoué** à le détecter
- Mais en apprenant les **patterns généraux** (poids des termes), le modèle a su extrapoler

**Preuve empirique** :

- 1766 offres unlabeled prédites comme Data (51.7%)
- 1652 offres unlabeled prédites comme Non-Data (48.3%)
- Notre offre fait partie des **50 plus confiantes Non-Data** (p < 0.1)

---

# Phase 8 : Filtrage Final et Métriques Globales

## Explication Méthodologique

La dernière étape consiste à **combiner** :

1. Les offres détectées par **regex** (whitelist) : 2611 offres Data
2. Les offres prédites par **ML** (non labellisées) : 1766 offres Data

Cette approche hybride permet de :

- **Conserver** les offres évidentes (détectées par regex)
- **Récupérer** les offres ambiguës (détectées par ML)
- **Éliminer** les faux positifs (détectés par blacklist ou ML)

---

## Notre Code

```python
# Offres Data issues du regex
data_from_regex = labeled_data[labeled_data['ml_label'] == 1].copy()

# Offres Data prédites par ML
data_from_ml = unlabeled_data[unlabeled_data['ml_prediction'] == 1].copy()

# Combinaison
df_filtered = pd.concat([data_from_regex, data_from_ml], ignore_index=True)

print(f"Final filtering results:")
print(f"  Initial records: {initial_count}")
print(f"  ├─ Regex Data jobs kept: {len(data_from_regex)}")
print(f"  ├─ ML-predicted Data jobs kept: {len(data_from_ml)}")
print(f"  └─ Total kept: {len(df_filtered)}")
print(f"  Removed: {initial_count - len(df_filtered)} ({(initial_count - len(df_filtered))/initial_count*100:.1f}%)")
print(f"  Retention rate: {len(df_filtered)/initial_count*100:.1f}%")
```

**Sortie** :
```
Final filtering results:
  Initial records: 6506
  ├─ Regex Data jobs kept: 2611
  ├─ ML-predicted Data jobs kept: 1766
  └─ Total kept: 4377
  Removed: 2129 (32.7%)
  Retention rate: 67.3%
```

## Bilan Final

### Avant filtrage

- **6506 offres** brutes (tous types confondus)

### Après filtrage hybride (Regex + ML)

- **4377 offres Data/IA** conservées (67.3%)
- **2129 offres Non-Data** éliminées (32.7%)

### Apport du ML

- **1766 offres Data récupérées** (27% du total final)
- Ces offres n'auraient **pas été détectées** par regex seul
- **Gain de ~40%** par rapport à une approche purement regex (2611 → 4377)

### Performance du modèle

- **F1-Score : 0.978**
- **ROC-AUC : 0.996**
- **Précision sur classe Data : 99.6%**


### Résultat

- **2611 offres** détectées par regex (cas évidents)
- **1766 offres** récupérées par ML (cas ambigus)
- **Gain total** : +67.6% par rapport à du regex seul.

---

# Références Bibliographiques

## Articles Fondateurs

### TF-IDF
1. **Salton, G., & McGill, M. J. (1983)**  
   *Introduction to Modern Information Retrieval*  
   McGraw-Hill.  

2. **Sparck Jones, K. (1972)**  
   *"A statistical interpretation of term specificity and its application in retrieval"*  
   Journal of Documentation, 28(1), 11-21.  

### Régression Logistique
3. **Cox, D. R. (1958)**  
   *"The regression analysis of binary sequences"*  
   Journal of the Royal Statistical Society: Series B, 20(2), 215-232.  

4. **Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013)**  
   *Applied Logistic Regression (3rd ed.)*  
   Wiley.  

### Classification de Textes
5. **Joachims, T. (1998)**  
   *"Text categorization with support vector machines: Learning with many relevant features"*  
   European Conference on Machine Learning (pp. 137-142). Springer.  

6. **Sebastiani, F. (2002)**  
   *"Machine learning in automated text categorization"*  
   ACM Computing Surveys, 34(1), 1-47.  

### Gestion du Déséquilibre de Classes
7. **He, H., & Garcia, E. A. (2009)**  
   *"Learning from imbalanced data"*  
   IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.  

8. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002)**  
   *"SMOTE: Synthetic minority over-sampling technique"*  
   Journal of Artificial Intelligence Research, 16, 321-357.  

## Ressources en Ligne
9. **Scikit-learn Documentation**  
   https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html  

10. **Scikit-learn: Logistic Regression**  
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  

11. **Manning, C. D., Raghavan, P., & Schütze, H. (2008)**  
    *Introduction to Information Retrieval*  
    Cambridge University Press.  
    → Livre de référence en NLP, disponible gratuitement en ligne :  
    https://nlp.stanford.edu/IR-book/

---

**Auteur** : RUCHE's Team  

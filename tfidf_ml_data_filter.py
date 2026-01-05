"""
ML-Based Job Filtering - TF-IDF + Logistic Regression
======================================================
Replaces regex-based whitelist/blacklist filtering with supervised ML

Advantages over regex:
- Learns complex patterns automatically
- Better generalization
- Easy to maintain (just add training examples)
- Interpretable coefficients
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')


def filter_data_jobs_ml(df: pd.DataFrame, 
                        french_stopwords: list,
                        test_size: float = 0.2,
                        random_state: int = 42) -> pd.DataFrame:
    """
    Filter job offers using ML instead of regex
    
    Strategy:
    ---------
    1. Use regex-matched offers as training data:
       - whitelist matches → label = 1 (Data job)
       - blacklist matches → label = 0 (Non-Data job)
    
    2. Train TF-IDF + Logistic Regression on title + description + job_function
    
    3. Predict on unmatched offers
    
    4. Remove offers predicted as Non-Data
    
    Args:
        df: DataFrame after regex filtering
        french_stopwords: List of French stopwords
        test_size: Proportion for validation
        random_state: Random seed
        
    Returns:
        Filtered DataFrame with ML predictions
    """
    
    print("\n" + "="*80)
    print("STEP 4: ML-BASED DATA JOB FILTERING")
    print("="*80)
    print("Strategy: TF-IDF + Logistic Regression")
    print(f"Replaces regex whitelist/blacklist with supervised learning")
    
    initial_count = len(df)
    
    # ========================================================================
    # PHASE 1: PREPARE TRAINING DATA FROM REGEX MATCHES
    # ========================================================================
    
    print("\n" + "─"*80)
    print("PHASE 1: TRAINING DATA PREPARATION")
    print("─"*80)
    
    # Apply regex patterns (same as before)
    whitelist_patterns = [
        r'\bdata\s*scientist\b', r'\bdata\s*analyst\b', r'\bdata\s*engineer\b',
        r'\bmachine\s+learning\b', r'\bml\s+engineer\b', r'\bdeep\s+learning\b',
        r'\bintelligence\s+artificielle\b', r'\bai\s+engineer\b',
        r'\bbig\s*data\b', r'\bhadoop\b', r'\bspark\b',
        r'\bbusiness\s+intelligence\b', r'\b\bbi\b.*\b(analyst|engineer|developer)\b',
        r'\banalytics\b', r'\bdata.*analytics\b',
        r'\bdata\s*architect\b', r'\barchitecte.*donn[ée]es\b',
        r'\bdata\s*manager\b', r'\bchef.*projet.*\bdata\b',
        r'\bstatisticien\b', r'\bbiostatisticien\b',
        r'\bnlp\s+engineer\b', r'\bnatural\s+language\b',
        r'\bcomputer\s+vision\b', r'\bvision.*ordinateur\b',
        r'\bmlops\b', r'\bdataops\b',
        r'\bpower\s*bi\b', r'\btableau\b', r'\bqlik\b', r'\bdatabricks\b'
    ]
    
    blacklist_patterns = [
        r'\binfirmier\b', r'\bm[ée]decin\b', r'\biade\b', r'\bibode\b',
        r'\bcadre\s+de\s+sant[ée]\b', r'\bsage[- ]femme\b',
        r'\bm[ée]canicien\b(?!.*\bdata\b)', r'\b[ée]lectricien\b', r'\bplombier\b',
        r'\btechnicien.*maintenance\b(?!.*(si|informatique|réseau))',
        r'\bsecr[ée]taire\b(?!.*(g[ée]n[ée]ral|direction))',
        r'\bassistant.*administratif\b(?!.*(si|data|informatique))',
        r'\benseignant\b(?!.*(data|ia|informatique))',
        r'\bformateur\b(?!.*(data|ia|python))',
        r'\bgestionnaire.*paie\b(?!.*(data|analytics|sirh))',
        r'\bcomptable\b(?!.*(data|analyste))',
        r'\bcommercial\b(?!.*(data|tech|software|saas))',
        r'\btechnico[- ]commercial\b(?!.*(data|it))',
        r'\bjuriste\b(?!.*(data|rgpd|ia|tech))',
        r'\bconducteur\b(?!.*(projet|travaux.*data))',
        r'\bchauffeur\b', r'\blivreur\b'
    ]
    
    # Create combined text
    df['combined_text'] = (
        df['title'].fillna('') + ' ' + 
        df['description'].fillna('') + ' ' +
        df['job_function'].fillna('')
    ).str.lower()
    
    # Apply patterns
    whitelist_mask = df['combined_text'].str.contains(
        '|'.join(whitelist_patterns), regex=True, case=False, na=False
    )
    blacklist_mask = df['combined_text'].str.contains(
        '|'.join(blacklist_patterns), regex=True, case=False, na=False
    )
    
    # Label data for training
    df['ml_label'] = None  # Will be filled
    df.loc[whitelist_mask & ~blacklist_mask, 'ml_label'] = 1  # Data job
    df.loc[blacklist_mask & ~whitelist_mask, 'ml_label'] = 0  # Non-Data job
    
    # Separate labeled (training) and unlabeled (to predict)
    labeled_data = df[df['ml_label'].notna()].copy()
    unlabeled_data = df[df['ml_label'].isna()].copy()
    
    print(f"\nData split:")
    print(f"  Total offers: {len(df)}")
    print(f"  ├─ Labeled (from regex): {len(labeled_data)} ({len(labeled_data)/len(df)*100:.1f}%)")
    print(f"  │  ├─ Data jobs (whitelist): {(labeled_data['ml_label']==1).sum()}")
    print(f"  │  └─ Non-Data jobs (blacklist): {(labeled_data['ml_label']==0).sum()}")
    print(f"  └─ Unlabeled (to predict): {len(unlabeled_data)} ({len(unlabeled_data)/len(df)*100:.1f}%)")
    
    # Check if we have enough training data
    if len(labeled_data) < 100:
        print("\n⚠️  WARNING: Not enough labeled data for ML!")
        print(f"   Only {len(labeled_data)} labeled examples. Falling back to regex only.")
        df_filtered = df[whitelist_mask & ~blacklist_mask].copy()
        return df_filtered.drop(['combined_text', 'ml_label'], axis=1, errors='ignore')
    
    # Check class balance
    class_counts = labeled_data['ml_label'].value_counts()
    balance_ratio = class_counts.min() / class_counts.max()
    print(f"\nClass balance:")
    print(f"  Ratio (minority/majority): {balance_ratio:.2f}")
    if balance_ratio < 0.3:
        print(f"  ⚠️  Imbalanced! Consider using class_weight='balanced'")
    
    # ========================================================================
    # PHASE 2: FEATURE ENGINEERING - TF-IDF
    # ========================================================================
    
    print("\n" + "─"*80)
    print("PHASE 2: FEATURE ENGINEERING")
    print("─"*80)
    
    print("\nTF-IDF Vectorization...")
    print(f"  Features: title + description + job_function")
    print(f"  Stopwords: {len(french_stopwords)} French stopwords")
    
    # TF-IDF on combined text
    vectorizer = TfidfVectorizer(
        max_features=500,           # Keep top 500 features
        max_df=0.7,                 # Ignore terms in >70% of docs
        min_df=2,                   # Ignore terms in <2 docs
        stop_words=french_stopwords,
        ngram_range=(1, 2),         # Unigrams + bigrams
        lowercase=True,
        strip_accents='unicode'
    )
    
    # Fit on labeled data only
    X_labeled = vectorizer.fit_transform(labeled_data['combined_text'])
    y_labeled = labeled_data['ml_label'].astype(int).values
    
    print(f"\n  ✅ TF-IDF matrix shape: {X_labeled.shape}")
    print(f"     Features extracted: {len(vectorizer.get_feature_names_out())}")
    print(f"     Sparsity: {(1 - X_labeled.nnz / np.prod(X_labeled.shape)) * 100:.1f}%")
    
    # ========================================================================
    # PHASE 3: MODEL TRAINING
    # ========================================================================
    
    print("\n" + "─"*80)
    print("PHASE 3: MODEL TRAINING")
    print("─"*80)
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_labeled, y_labeled, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_labeled
    )
    
    print(f"\nTrain/Validation split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"    ├─ Data jobs: {(y_train==1).sum()}")
    print(f"    └─ Non-Data jobs: {(y_train==0).sum()}")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"    ├─ Data jobs: {(y_val==1).sum()}")
    print(f"    └─ Non-Data jobs: {(y_val==0).sum()}")
    
    # Train Logistic Regression
    print(f"\nTraining Logistic Regression...")
    
    model = LogisticRegression(
        class_weight='balanced',    # Handle class imbalance
        max_iter=1000,
        random_state=random_state,
        solver='liblinear'          # Good for small datasets
    )
    
    model.fit(X_train, y_train)
    
    print(f"  ✅ Model trained successfully!")
    
    # Cross-validation
    print(f"\n5-Fold Cross-Validation:")
    cv_scores = cross_val_score(model, X_labeled, y_labeled, cv=5, scoring='f1')
    print(f"  F1 scores: {cv_scores}")
    print(f"  Mean F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # ========================================================================
    # PHASE 4: MODEL EVALUATION
    # ========================================================================
    
    print("\n" + "─"*80)
    print("PHASE 4: MODEL EVALUATION")
    print("─"*80)
    
    # Predictions on validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Classification report
    print("\nClassification Report (Validation Set):")
    print(classification_report(
        y_val, y_val_pred, 
        target_names=['Non-Data', 'Data'],
        digits=3
    ))
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_val, y_val_proba)
    print(f"ROC-AUC Score: {roc_auc:.3f}")
    
    # ========================================================================
    # PHASE 5: MODEL INTERPRETABILITY
    # ========================================================================
    
    print("\n" + "─"*80)
    print("PHASE 5: MODEL INTERPRETABILITY")
    print("─"*80)
    
    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Sort by absolute value
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    # Top positive features (Data job indicators)
    print("\n🔵 Top 15 POSITIVE features (Data job indicators):")
    top_positive = feature_importance.nlargest(15, 'coefficient')
    for idx, row in top_positive.iterrows():
        print(f"  {row['feature']:30s} → {row['coefficient']:+.3f}")
    
    # Top negative features (Non-Data job indicators)
    print("\n🔴 Top 15 NEGATIVE features (Non-Data job indicators):")
    top_negative = feature_importance.nsmallest(15, 'coefficient')
    for idx, row in top_negative.iterrows():
        print(f"  {row['feature']:30s} → {row['coefficient']:+.3f}")
    
    # ========================================================================
    # PHASE 6: VISUALIZATIONS
    # ========================================================================
    
    print("\n" + "─"*80)
    print("PHASE 6: VISUALIZATIONS")
    print("─"*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['Non-Data', 'Data'],
        yticklabels=['Non-Data', 'Data'],
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Confusion Matrix (Validation Set)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. Top Features
    top_features = pd.concat([
        feature_importance.nlargest(10, 'coefficient'),
        feature_importance.nsmallest(10, 'coefficient')
    ])
    
    colors = ['green' if c > 0 else 'red' for c in top_features['coefficient']]
    axes[0, 1].barh(range(len(top_features)), top_features['coefficient'], color=colors)
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['feature'], fontsize=9)
    axes[0, 1].set_xlabel('Coefficient', fontsize=11)
    axes[0, 1].set_title('Top 20 Features by Coefficient', fontsize=14, fontweight='bold')
    axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_val_proba)
    axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[1, 0].set_xlabel('False Positive Rate', fontsize=11)
    axes[1, 0].set_ylabel('True Positive Rate', fontsize=11)
    axes[1, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Probability Distribution
    proba_data = y_val_proba[y_val == 1]
    proba_non_data = y_val_proba[y_val == 0]
    
    axes[1, 1].hist(proba_non_data, bins=30, alpha=0.6, label='Non-Data', color='red', edgecolor='black')
    axes[1, 1].hist(proba_data, bins=30, alpha=0.6, label='Data', color='green', edgecolor='black')
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')
    axes[1, 1].set_xlabel('Predicted Probability', fontsize=11)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title('Predicted Probability Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_job_filtering_analysis.png', dpi=300, bbox_inches='tight')
    print("\n  ✅ Visualizations saved: ml_job_filtering_analysis.png")
    plt.close()
    
    # ========================================================================
    # PHASE 7: PREDICTION ON UNLABELED DATA
    # ========================================================================
    
    print("\n" + "─"*80)
    print("PHASE 7: PREDICTION ON UNLABELED DATA")
    print("─"*80)
    
    if len(unlabeled_data) > 0:
        print(f"\nPredicting on {len(unlabeled_data)} unlabeled offers...")
        
        # Transform unlabeled data
        X_unlabeled = vectorizer.transform(unlabeled_data['combined_text'])
        
        # Predict
        unlabeled_pred = model.predict(X_unlabeled)
        unlabeled_proba = model.predict_proba(X_unlabeled)[:, 1]
        
        # Add predictions to unlabeled data
        unlabeled_data['ml_prediction'] = unlabeled_pred
        unlabeled_data['ml_probability'] = unlabeled_proba
        
        print(f"\nPrediction results:")
        print(f"  Predicted as Data: {(unlabeled_pred==1).sum()} ({(unlabeled_pred==1).sum()/len(unlabeled_data)*100:.1f}%)")
        print(f"  Predicted as Non-Data: {(unlabeled_pred==0).sum()} ({(unlabeled_pred==0).sum()/len(unlabeled_data)*100:.1f}%)")
        
        print(f"\nProbability statistics:")
        print(f"  Mean probability (Data class): {unlabeled_proba.mean():.3f}")
        print(f"  Median probability: {np.median(unlabeled_proba):.3f}")
        print(f"  Std probability: {unlabeled_proba.std():.3f}")
        
        # Show examples of predictions
        print(f"\n📝 Sample predictions (High confidence Data jobs):")
        high_conf_data = unlabeled_data[unlabeled_proba > 0.9].head(3)
        for idx, row in high_conf_data.iterrows():
            print(f"  • {row['title'][:60]:60s} (p={row['ml_probability']:.3f})")
        
        print(f"\n📝 Sample predictions (High confidence Non-Data jobs):")
        high_conf_non_data = unlabeled_data[unlabeled_proba < 0.1].head(3)
        for idx, row in high_conf_non_data.iterrows():
            print(f"  • {row['title'][:60]:60s} (p={row['ml_probability']:.3f})")
    else:
        print("\n  No unlabeled data to predict on.")
        unlabeled_data['ml_prediction'] = None
        unlabeled_data['ml_probability'] = None
    
    # ========================================================================
    # PHASE 8: FINAL FILTERING
    # ========================================================================
    
    print("\n" + "─"*80)
    print("PHASE 8: FINAL FILTERING")
    print("─"*80)
    
    # Keep:
    # - All regex-labeled Data jobs (ml_label == 1)
    # - ML-predicted Data jobs from unlabeled (ml_prediction == 1)
    
    # Regex-labeled Data jobs
    data_from_regex = labeled_data[labeled_data['ml_label'] == 1].copy()
    
    # ML-predicted Data jobs
    data_from_ml = unlabeled_data[unlabeled_data['ml_prediction'] == 1].copy() if len(unlabeled_data) > 0 else pd.DataFrame()
    
    # Combine
    df_filtered = pd.concat([data_from_regex, data_from_ml], ignore_index=True)
    
    # Clean up temporary columns
    df_filtered = df_filtered.drop(['combined_text', 'ml_label', 'ml_prediction', 'ml_probability'], axis=1, errors='ignore')
    
    print(f"\nFinal filtering results:")
    print(f"  Initial records: {initial_count}")
    print(f"  ├─ Regex Data jobs kept: {len(data_from_regex)}")
    print(f"  ├─ ML-predicted Data jobs kept: {len(data_from_ml)}")
    print(f"  └─ Total kept: {len(df_filtered)}")
    print(f"  Removed: {initial_count - len(df_filtered)} ({(initial_count - len(df_filtered))/initial_count*100:.1f}%)")
    print(f"  Retention rate: {len(df_filtered)/initial_count*100:.1f}%")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("ML FILTERING COMPLETE")
    print("="*80)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  • Validation F1-Score: {cv_scores.mean():.3f}")
    print(f"  • ROC-AUC: {roc_auc:.3f}")
    print(f"  • Final retention: {len(df_filtered)/initial_count*100:.1f}%")
    
    print(f"\n💾 Outputs:")
    print(f"  • Visualization: ml_job_filtering_analysis.png")
    print(f"  • Filtered dataset: {len(df_filtered)} records")
    
    print(f"\n✨ Model ready for deployment!")
    
    return df_filtered

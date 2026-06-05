import glob
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import EXPERIMENTS_DIR, MODELS_DIR, PROCESSED_DIR
from perestruct_classifier_train.model_title import ArticleLinkerTitlesYOLO
from perestruct_classifier_train.preprocess import vectorize_texts_only

# Configuration for Titles Model
CONFIG = {
    'vectorizer': {
        'analyzer': 'char_wb',
        'ngram_range': (2, 4),
        'max_features': 15000,
        'min_df': 3,
        'max_df': 0.85,
    },
    'model': {
        'n_estimators': 400,
        'max_depth': 10,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_jobs': -1,
        'min_child_weight': 2
    },
    'yolo': {
        'dim': 576,
    }
}


def find_titles_dataset():
    """Find dataset file with features for title model."""
    pattern = str(PROCESSED_DIR / "pairs_soviet_title.pkl")
    files = glob.glob(pattern)

    if not files:
        fallback = PROCESSED_DIR / "pairs_soviet_title.pkl"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            f"YOLO feature files for titles not found: {pattern}. "
            f"Run prepare_dataset_titles_yolo.py first."
        )

    latest_file = max(files, key=os.path.getmtime)
    return Path(latest_file)


def train_titles_yolo_model():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAINING TITLES MODEL (Title-Plain + Title-Title + YOLO)")
    print("=" * 60)
    print(f"Model parameters: {CONFIG['model']}")
    print(f"YOLO dimension: {CONFIG['yolo']['dim']}")
    print("Additional features: block types (is_title_1, is_title_2)")
    print("=" * 60)

    # Load data
    print("\nSearching for title dataset...")
    pairs_path = find_titles_dataset()
    print(f"   Found file: {pairs_path.name}")

    with open(pairs_path, 'rb') as f:
        pairs_raw = pickle.load(f)

    # Check for required features
    sample_pair = pairs_raw['train_pos'][0] if pairs_raw['train_pos'] else {}
    has_yolo = 'yolo_emb' in sample_pair and sample_pair['yolo_emb'] is not None
    has_types = 'type1' in sample_pair and 'type2' in sample_pair

    if not has_yolo:
        print("   ERROR: No 'yolo_emb' key found in data!")
        return
    if not has_types:
        print("   WARNING: No 'type1'/'type2' keys found. "
              "Model will be less effective.")

    print("   Pre-computed YOLO vectors detected.")
    if has_types:
        print("   Block type features detected.")

    # Collect texts for TF-IDF
    all_texts = []
    counts = {'train_pos': 0, 'train_neg': 0}
    type_stats = {}

    for key in ['train_pos', 'train_neg']:
        if key in pairs_raw:
            counts[key] = len(pairs_raw[key])
            for pair in pairs_raw[key]:
                all_texts.append(pair['text1'])
                all_texts.append(pair['text2'])

                # Type statistics
                if has_types:
                    t1, t2 = pair.get('type1'), pair.get('type2')
                    pair_type = f"{t1} ↔ {t2}"
                    type_stats[pair_type] = type_stats.get(pair_type, 0) + 1

    print(f"   Positive pairs: {counts['train_pos']}")
    print(f"   Negative pairs: {counts['train_neg']}")
    print(f"   Texts for TF-IDF: {len(all_texts)}")
    if type_stats:
        print(f"   Pair type distribution: {type_stats}")

    if len(all_texts) == 0:
        raise ValueError("No texts for training! Check dataset.")

    # Train vectorizer
    print("\nTraining TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer=CONFIG['vectorizer']['analyzer'],
        ngram_range=CONFIG['vectorizer']['ngram_range'],
        max_features=CONFIG['vectorizer']['max_features'],
        min_df=CONFIG['vectorizer']['min_df'],
        max_df=CONFIG['vectorizer']['max_df'],
        dtype=np.float32
    )
    vectorizer.fit(all_texts)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"   Vocabulary size: {vocab_size}")

    # Vectorize texts
    print("\nAdding TF-IDF vectors to pairs...")
    pairs_vectorized = vectorize_texts_only(pairs_raw, vectorizer)

    if not pairs_vectorized or 'train_pos' not in pairs_vectorized:
        raise ValueError("Vectorization error: result empty.")

    # Train model
    print("\nTraining ArticleLinkerTitlesYOLO model...")
    model = ArticleLinkerTitlesYOLO(
        vocab_size=vocab_size,
        yolo_dim=CONFIG['yolo']['dim']
    )

    model.fit(
        train_pos=pairs_vectorized['train_pos'],
        train_neg=pairs_vectorized['train_neg'],
        val_pos=pairs_vectorized['val_pos'],
        val_neg=pairs_vectorized['val_neg'],
        model_params=CONFIG['model']
    )

    # Evaluate and save
    print("\nEvaluating on test set...")
    metrics = model.evaluate(
        test_pos=pairs_vectorized['test_pos'],
        test_neg=pairs_vectorized['test_neg']
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"model_titles_yolo_auc{metrics['auc']:.4f}_{timestamp}.pkl"
    model_path = MODELS_DIR / model_filename
    model.save(model_path)

    vec_filename = f"vectorizer_titles_yolo_{timestamp}.pkl"
    vec_path = MODELS_DIR / vec_filename
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"   Vectorizer saved: {vec_path}")

    metadata = {
        'timestamp': timestamp,
        'config': CONFIG,
        'vocab_size': vocab_size,
        'metrics': metrics,
        'source_dataset': pairs_path.name,
        'feature_breakdown': {
            'tfidf_per_block': vocab_size,
            'coords': 8,
            'geo_features': 20,
            'type_features': 2,
            'yolo_features': CONFIG['yolo']['dim'],
            'total': 2 * vocab_size + 30 + CONFIG['yolo']['dim']
        }
    }

    meta_path = MODELS_DIR / f"metadata_titles_yolo_{timestamp}.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   Metadata saved: {meta_path}")

    # Copy to best files
    best_model_link = MODELS_DIR / "best_soviet_title_model.pkl"
    best_vec_link = MODELS_DIR / "best_soviet_title_vectorizer.pkl"
    best_meta_link = MODELS_DIR / "best_soviet_title_model_metadata.json"

    shutil.copy(model_path, best_model_link)
    shutil.copy(vec_path, best_vec_link)
    shutil.copy(meta_path, best_meta_link)

    print("\n" + "=" * 60)
    print("TITLES MODEL TRAINING COMPLETE!")
    print(f"   AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")
    print(f"   Files saved to: {MODELS_DIR.absolute()}")
    print("   - best_soviet_title_model.pkl")
    print("   - best_soviet_title_vectorizer.pkl")
    print("=" * 60)


if __name__ == '__main__':
    train_titles_yolo_model()

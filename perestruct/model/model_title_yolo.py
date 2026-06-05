import pickle

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm


class ArticleLinkerTitlesYOLO:
    def __init__(self, vocab_size, yolo_dim=576):
        self.vocab_size = vocab_size
        self.yolo_dim = yolo_dim
        self.model = None
        # Total dimension: TF-IDF(2*N) + Coords(8) + Geo(20) + Types(2) + YOLO(576)
        # Total: 2*N + 606 features
        self.total_features = 2 * vocab_size + 8 + 20 + 2 + yolo_dim

    def _get_geo_features(self, c1, c2):
        """Calculate 20 geometric features between two boxes."""
        x1, y1, x1b, y1b = c1
        x2, y2, x2b, y2b = c2

        w1, h1 = x1b - x1, y1b - y1
        w2, h2 = x2b - x2, y2b - y2

        cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
        cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2

        dx, dy = cx1 - cx2, cy1 - cy2

        return np.array([
            np.sqrt(dx ** 2 + dy ** 2),
            abs(dx) + abs(dy),
            max(abs(dx), abs(dy)),
            abs(dx),
            abs(dy),
            float(cy1 < cy2),
            float(cx1 < cx2),
            float(cx1 > cx2) * 2 + float(cy1 > cy2),
            float(abs(dx) < min(w1, w2) / 2),
            float(abs(dy) < min(h1, h2) / 2),
            w1,
            h1,
            w2,
            h2,
            w1 / (w2 + 1e-8),
            h1 / (h2 + 1e-8),
            abs(w1 * h1 - w2 * h2),
            (w1 * h1) / (w2 * h2 + 1e-8),
            max(0, x2 - x1b) if x2 > x1b else max(0, x1 - x2b),
            max(0, y2 - y1b) if y2 > y1b else max(0, y1 - y2b),
        ], dtype=np.float16)

    def _get_type_features(self, type1, type2):
        """2 binary features indicating block types."""
        t1 = str(type1).lower().strip()
        t2 = str(type2).lower().strip()

        is_title_1 = 1.0 if t1 == 'title' else 0.0
        is_title_2 = 1.0 if t2 == 'title' else 0.0

        return np.array([is_title_1, is_title_2], dtype=np.float16)

    def _get_features(
        self,
        vec1,
        c1,
        vec2,
        c2,
        yolo_emb=None,
        type1='plain text',
        type2='plain text'
    ):
        """Build feature vector: TF-IDF + Coords + Geo + Types + YOLO."""
        d1 = vec1.toarray().flatten()
        d2 = vec2.toarray().flatten()

        coords = np.array([
            c1[0], c1[1], c1[2], c1[3],
            c2[0], c2[1], c2[2], c2[3]
        ], dtype=np.float16)

        geo = self._get_geo_features(c1, c2)
        types = self._get_type_features(type1, type2)

        if yolo_emb is None:
            yolo_vec = np.zeros(self.yolo_dim, dtype=np.float16)
        else:
            yolo_vec = np.array(yolo_emb, dtype=np.float16).flatten()
            if yolo_vec.shape[0] != self.yolo_dim:
                yolo_vec = np.zeros(self.yolo_dim, dtype=np.float16)

        return np.concatenate([d1, d2, coords, geo, types, yolo_vec]).astype(
            np.float16
        )

    def _prepare(self, pairs, labels):
        print(f"Preparing {len(pairs)} pairs with Type Features...")
        X = []

        missing_yolo_count = 0
        type_stats = {}

        for item in tqdm(pairs, desc="Extracting Features"):
            if isinstance(item, dict):
                v1, c1 = item['vec1'], item['c1']
                v2, c2 = item['vec2'], item['c2']
                t1 = item.get('type1', 'plain text')
                t2 = item.get('type2', 'plain text')

                pair_type = f"{t1} ↔ {t2}"
                type_stats[pair_type] = type_stats.get(pair_type, 0) + 1

                yolo_emb = item.get('yolo_emb', None)
                if yolo_emb is None:
                    missing_yolo_count += 1
            else:
                raise ValueError("Expected dictionary")

            X.append(
                self._get_features(
                    v1, c1, v2, c2, yolo_emb=yolo_emb, type1=t1, type2=t2
                )
            )

        if missing_yolo_count > 0:
            print(f"   Warning: {missing_yolo_count} pairs had no YOLO embedding.")

        print(f"   Pair types in dataset: {type_stats}")

        return np.array(X, dtype=np.float16), np.array(labels, dtype=np.float16)

    def fit(self, train_pos, train_neg, val_pos, val_neg, model_params=None):
        """Train the model for title linking."""
        print("Training Titles Model (Title-Plain / Title-Title)")
        print("-" * 60)

        X_tr, y_tr = self._prepare(
            train_pos + train_neg,
            [1] * len(train_pos) + [0] * len(train_neg)
        )
        X_val, y_val = self._prepare(
            val_pos + val_neg,
            [1] * len(val_pos) + [0] * len(val_neg)
        )

        print(f"\nMatrix shape: {X_tr.shape}")
        print(f"   Expected Total Features: {self.total_features}")
        print(f"   - TF-IDF: {2 * self.vocab_size}")
        print(f"   - Coords: 8")
        print(f"   - Geo: 20")
        print(f"   - Types: 2 (is_title_1, is_title_2)")
        print(f"   - YOLO: {self.yolo_dim}")

        if X_tr.shape[1] != self.total_features:
            print(
                f"Error: Feature dimension mismatch! "
                f"Got {X_tr.shape[1]}, expected {self.total_features}"
            )
            return self

        if model_params is None:
            model_params = {
                'n_estimators': 400,
                'max_depth': 10,
                'learning_rate': 0.02,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_jobs': -1,
                'random_state': 42,
                'min_child_weight': 2,
            }

        scale_pos_weight = len(train_neg) / max(len(train_pos), 1)

        self.model = xgb.XGBClassifier(
            **model_params,
            scale_pos_weight=scale_pos_weight
        )

        print(f"Training with params: {model_params}")
        print(f"Train samples: {len(X_tr)}")
        print("\nTraining...")

        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=True)

        return self

    def predict(self, vec1, c1, vec2, c2, yolo_emb=None, type1='plain text',
                type2='plain text'):
        """Predict probability for a single pair."""
        features = self._get_features(
            vec1, c1, vec2, c2, yolo_emb=yolo_emb, type1=type1, type2=type2
        )
        return float(self.model.predict_proba(features.reshape(1, -1))[0][1])

    def predict_batch(self, pairs):
        """Batch prediction for multiple pairs."""
        if not pairs:
            return np.array([])

        X = []
        for item in tqdm(pairs, desc="Predicting"):
            if isinstance(item, dict):
                v1, c1 = item['vec1'], item['c1']
                v2, c2 = item['vec2'], item['c2']
                yolo_emb = item.get('yolo_emb', None)
                t1 = item.get('type1', 'plain text')
                t2 = item.get('type2', 'plain text')
            else:
                raise ValueError("Expected dictionary")

            X.append(
                self._get_features(
                    v1, c1, v2, c2, yolo_emb=yolo_emb, type1=t1, type2=t2
                )
            )

        return self.model.predict_proba(np.array(X, dtype=np.float16))[:, 1]

    def evaluate(self, test_pos, test_neg):
        """Evaluate model performance on test data."""
        print("\nEvaluating Titles Model on test set...")
        X_te, y_te = self._prepare(
            test_pos + test_neg,
            [1] * len(test_pos) + [0] * len(test_neg)
        )
        preds = self.model.predict_proba(X_te)[:, 1]
        preds_cls = (preds >= 0.5).astype(int)

        auc = roc_auc_score(y_te, preds)
        acc = accuracy_score(y_te, preds_cls)
        f1 = f1_score(y_te, preds_cls)

        print(f"\nRESULTS (Titles): AUC={auc:.4f} | Acc={acc:.4f} | F1={f1:.4f}")
        return {'auc': auc, 'f1': f1, 'accuracy': acc}

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved: {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded: {path}")
        return self

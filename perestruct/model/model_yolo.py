import pickle

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm


class ArticleLinkerYOLO:
    def __init__(self, vocab_size, yolo_dim=576):
        self.vocab_size = vocab_size
        self.yolo_dim = yolo_dim
        self.model = None
        # Total dimension: TF-IDF(2*N) + Coords(8) + Geo(20) + YOLO(576)
        self.total_features = 2 * vocab_size + 8 + 20 + yolo_dim

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

    def _get_features(self, vec1, c1, vec2, c2, yolo_emb=None):
        """Build feature vector: TF-IDF + Coords + Geo + YOLO."""
        d1 = vec1.toarray().flatten()
        d2 = vec2.toarray().flatten()

        coords = np.array([
            c1[0], c1[1], c1[2], c1[3],
            c2[0], c2[1], c2[2], c2[3]
        ], dtype=np.float16)

        geo = self._get_geo_features(c1, c2)

        if yolo_emb is None:
            yolo_vec = np.zeros(self.yolo_dim, dtype=np.float16)
        else:
            yolo_vec = np.array(yolo_emb, dtype=np.float16).flatten()

        return np.concatenate([d1, d2, coords, geo, yolo_vec]).astype(
            np.float16
        )

    def _prepare(self, pairs, labels):
        print(f"Preparing {len(pairs)} pairs with YOLO features...")
        X = []

        missing_yolo_count = 0

        for item in tqdm(pairs, desc="Extracting Features"):
            if isinstance(item, dict):
                v1, c1 = item['vec1'], item['c1']
                v2, c2 = item['vec2'], item['c2']
                yolo_emb = item.get('yolo_emb', None)
                if yolo_emb is None:
                    missing_yolo_count += 1
            else:
                raise ValueError(
                    "Expected dictionary with keys vec1, c1, vec2, c2, "
                    "[yolo_emb]"
                )

            X.append(self._get_features(v1, c1, v2, c2, yolo_emb=yolo_emb))

        if missing_yolo_count > 0:
            print(
                f"   Warning: {missing_yolo_count} pairs had no YOLO "
                f"embedding (filled with zeros)."
            )

        return np.array(X, dtype=np.float16), np.array(labels, dtype=np.float16)

    def fit(self, train_pos, train_neg, val_pos, val_neg, model_params=None):
        """Train the model."""
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
        print(f"   - YOLO: {self.yolo_dim}")

        if X_tr.shape[1] != self.total_features:
            print(
                f"Error: Feature dimension mismatch! Got {X_tr.shape[1]}, "
                f"expected {self.total_features}"
            )
            return self

        if model_params is None:
            model_params = {
                'n_estimators': 300,
                'max_depth': 12,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_jobs': -1,
                'random_state': 42,
            }

        scale_pos_weight = len(train_neg) / max(len(train_pos), 1)

        self.model = xgb.XGBClassifier(
            **model_params,
            scale_pos_weight=scale_pos_weight
        )

        print(f"Training with params: {model_params}")
        print("\nTraining...")
        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=True)

        return self

    def predict(self, vec1, c1, vec2, c2, yolo_emb=None):
        """Predict probability for a single pair."""
        features = self._get_features(vec1, c1, vec2, c2, yolo_emb=yolo_emb)
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
            else:
                raise ValueError("Expected dictionary")

            X.append(self._get_features(v1, c1, v2, c2, yolo_emb=yolo_emb))

        return self.model.predict_proba(np.array(X, dtype=np.float16))[:, 1]

    def evaluate(self, test_pos, test_neg):
        """Evaluate model performance on test data."""
        X_te, y_te = self._prepare(
            test_pos + test_neg,
            [1] * len(test_pos) + [0] * len(test_neg)
        )
        preds = self.model.predict_proba(X_te)[:, 1]
        preds_cls = (preds >= 0.5).astype(int)

        auc = roc_auc_score(y_te, preds)
        acc = accuracy_score(y_te, preds_cls)
        f1 = f1_score(y_te, preds_cls)

        print(f"\nRESULTS: AUC={auc:.4f} | Acc={acc:.4f} | F1={f1:.4f}")
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

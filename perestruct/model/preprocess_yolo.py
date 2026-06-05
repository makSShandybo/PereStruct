import string
from pathlib import Path

import cv2
import nltk
import numpy as np
import pymorphy3
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

_stop_words = set(stopwords.words('russian'))
_punctuation = set(string.punctuation)
_morph = pymorphy3.MorphAnalyzer()


class YoloEmbedder:
    """Singleton class for loading and using YOLO for embeddings."""
    _instance = None
    _model_wrapper = None
    _device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _img_cache = {}

    @classmethod
    def get_instance(cls, weights_path='soviet_yolo.pt'):
        if cls._instance is None:
            cls._instance = cls()
            if weights_path:
                cls._instance.load_model(weights_path)
        return cls._instance

    def load_model(self, weights_path):
        print(f"Loading YOLO for visual embeddings: {weights_path}...")
        try:
            from doclayout_yolo.nn.tasks import attempt_load_one_weight
            import torch.serialization
            try:
                from doclayout_yolo.nn.tasks import YOLOv10DetectionModel
                from torch.nn.modules.container import Sequential
                torch.serialization.add_safe_globals([
                    YOLOv10DetectionModel,
                    Sequential
                ])
            except Exception:
                pass

            self._model_wrapper, _ = attempt_load_one_weight(weights_path)
            self._model_wrapper.to(self._device)
            print(f'Device: {self._device}')
            self._model_wrapper.eval()
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO: {e}")
            self._model_wrapper = None

    def _get_bbox_crop(self, img, c1, c2):
        """
        Get bounding box around two blocks.
        c1, c2 = (x1, y1, x2, y2) - normalized coordinates (0..1).
        """
        h, w = img.shape[:2]

        x1_1, y1_1, x2_1, y2_1 = int(c1[0] * w), int(c1[1] * h), int(c1[2] * w), int(c1[3] * h)
        x1_2, y1_2, x2_2, y2_2 = int(c2[0] * w), int(c2[1] * h), int(c2[2] * w), int(c2[3] * h)

        x_min = max(0, min(x1_1, x1_2))
        y_min = max(0, min(y1_1, y1_2))
        x_max = min(w, max(x2_1, x2_2))
        y_max = min(h, max(y2_1, y2_2))

        pad = 5
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)

        return img[y_min:y_max, x_min:x_max]

    def get_embedding(self, img_path, c1, c2, imgsz=1024):
        """Get embedding from bounding box of two blocks."""
        if self._model_wrapper is None:
            return np.zeros(576)

        path_str = str(img_path)
        if path_str not in self._img_cache:
            img = cv2.imread(path_str)
            if img is None:
                return np.zeros(576)
            self._img_cache[path_str] = img
        else:
            img = self._img_cache[path_str]

        roi = self._get_bbox_crop(img, c1, c2)
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return np.zeros(576)

        roi_resized = cv2.resize(roi, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

        im = roi_resized[..., ::-1].transpose((2, 0, 1))
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self._device).float() / 255.0
        im = im.unsqueeze(0)

        with torch.no_grad():
            try:
                preds = self._model_wrapper(
                    im,
                    augment=False,
                    visualize=False,
                    embed=[22]
                )

                tensor_data = None
                if isinstance(preds, tuple):
                    val = preds[0]
                    if isinstance(val, dict):
                        tensor_data = val.get(22) or list(val.values())[0]
                    elif isinstance(val, (tuple, list)):
                        tensor_data = val[0]
                    else:
                        tensor_data = val
                elif isinstance(preds, dict):
                    val = preds.get(22)
                    if val is None:
                        val = list(preds.values())[0]
                    if isinstance(val, (tuple, list)):
                        tensor_data = val[0]
                    else:
                        tensor_data = val
                elif isinstance(preds, list):
                    val = preds[0]
                    if isinstance(val, (tuple, list)):
                        val = val[0]
                    tensor_data = val
                elif isinstance(preds, torch.Tensor):
                    tensor_data = preds

                if tensor_data is None:
                    return np.zeros(576)

                if tensor_data.dim() == 4:
                    final = tensor_data.mean(dim=(2, 3))
                elif tensor_data.dim() == 3:
                    final = tensor_data.mean(dim=2)
                else:
                    final = tensor_data

                return final.squeeze(0).cpu().numpy()

            except Exception as e:
                print(f"Inference error: {e}")
                return np.zeros(576)


def lemmatize(tokens):
    """Lemmatize list of tokens."""
    result = []
    for word in tokens:
        if word.isalpha():
            result.append(_morph.parse(word)[0].normal_form)
        else:
            result.append(word)
    return ' '.join(result)


def preprocess_text(text):
    """Full text preprocessing."""
    if not text:
        return ""
    words = word_tokenize(text.lower())
    filtered = [
        w for w in words
        if w not in _stop_words and w not in _punctuation
    ]
    return lemmatize(filtered)


def get_block_type_code(box_type):
    """Convert block type name to code (0 or 1)."""
    block_type_map = {
        'plain text': 0,
        'title': 1
    }
    return block_type_map.get(box_type, 0)


def vectorize_pairs(pairs_dict, vectorizer, yolo_weights_path=None):
    """
    Vectorize pairs.
    Filter: processes only 'plain text' to 'plain text' pairs.
    If yolo_weights_path is provided, computes visual embedding (576).
    Returns: vec1, c1, vec2, c2, img_path, img_size, yolo_emb.
    """
    embedder = None
    if yolo_weights_path and Path(yolo_weights_path).exists():
        embedder = YoloEmbedder.get_instance(yolo_weights_path)
        if embedder._model_wrapper is None:
            print("YOLO not loaded, using zero vectors.")
    elif yolo_weights_path:
        print(f"YOLO weights file not found: {yolo_weights_path}")

    result = {}

    for key in ['train_pos', 'train_neg', 'val_pos', 'val_neg',
                'test_pos', 'test_neg']:
        pairs = pairs_dict.get(key, [])
        vectorized = []

        skipped_count = 0

        for pair in tqdm(pairs, desc=f"Vectorizing {key}"):
            type1_str = pair.get('type1', 'plain text')
            type2_str = pair.get('type2', 'plain text')

            if type1_str != 'plain text' or type2_str != 'plain text':
                skipped_count += 1
                continue

            vec1 = vectorizer.transform([pair['text1']])
            vec2 = vectorizer.transform([pair['text2']])

            yolo_emb = np.zeros(576, dtype=np.float32)
            if embedder and embedder._model_wrapper is not None:
                img_path = pair.get('img_path')
                c1 = pair.get('c1')
                c2 = pair.get('c2')
                if img_path and c1 and c2:
                    yolo_emb = embedder.get_embedding(img_path, c1, c2)

            vectorized.append({
                'vec1': vec1,
                'c1': pair['c1'],
                'vec2': vec2,
                'c2': pair['c2'],
                'img_path': pair['img_path'],
                'img_size': pair['img_size'],
                'yolo_emb': yolo_emb,
            })

        if skipped_count > 0:
            print(f"   Skipped {skipped_count} pairs containing title in {key}")
        if embedder:
            print(f"   {key}: Processed {len(vectorized)} pairs with YOLO")
        else:
            print(f"   {key}: Processed {len(vectorized)} pairs (text only)")

        result[key] = vectorized

    return result


def vectorize_texts_only(pairs_dict, vectorizer):
    """
    Quick text vectorization for pairs that already have yolo_emb.
    Does not filter types (assumes data is already clean).
    Does not recompute YOLO (uses existing from pair).
    Returns all split keys.
    """
    result = {}
    keys_to_process = [
        'train_pos', 'train_neg', 'val_pos', 'val_neg',
        'test_pos', 'test_neg'
    ]

    print("Vectorizing texts (TF-IDF)...")

    for key in keys_to_process:
        result[key] = []
        pairs = pairs_dict.get(key, [])
        if not pairs:
            continue

        vectorized = []

        for pair in tqdm(pairs, desc=f"   {key}"):
            vec1 = vectorizer.transform([pair['text1']])
            vec2 = vectorizer.transform([pair['text2']])

            yolo_emb = pair.get('yolo_emb')
            yolo_emb = np.array(yolo_emb, dtype=np.float32)

            vectorized.append({
                'vec1': vec1,
                'c1': pair['c1'],
                'vec2': vec2,
                'c2': pair['c2'],
                'img_path': pair.get('img_path'),
                'img_size': pair.get('img_size'),
                'yolo_emb': yolo_emb,
                'type1': pair.get('type1'),
                'type2': pair.get('type2'),
            })

        result[key] = vectorized
        print(f"   {key}: {len(vectorized)} pairs processed")

    return result

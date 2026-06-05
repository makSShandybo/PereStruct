import json
import os
import pickle
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "model"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model_yolo import ArticleLinkerYOLO
from model_title_yolo import ArticleLinkerTitlesYOLO
from preprocess_yolo import YoloEmbedder, preprocess_text



class ArticleAssembler:
    """Assembles text blocks into articles using ML models and spatial clustering."""

    def __init__(self, threshold=0.5, yolo_weights_name="soviet_yolo.pt"):
        self.threshold = threshold
        self.model_plain = None
        self.vec_plain = None
        self.model_titles = None
        self.vec_titles = None
        self.yolo_embedder = None

        self._load_models(yolo_weights_name)

    def _load_models(self, yolo_weights_name):
        """Load all required ML models and vectorizers from the model directory."""
        print("Loading models from 'model' directory...")

        files = {
            "plain_model": MODEL_DIR / "best_soviet_model.pkl",
            "plain_vec": MODEL_DIR / "best_soviet_vectorizer.pkl",
            "title_model": MODEL_DIR / "best_soviet_title_model.pkl",
            "title_vec": MODEL_DIR / "best_soviet_title_vectorizer.pkl",
            "plain_meta": MODEL_DIR / "best_soviet_model_metadata.json",
            "title_meta": MODEL_DIR / "best_soviet_title_model_metadata.json",
        }

        for name, path in files.items():
            if not path.exists() and "meta" not in name:
                raise FileNotFoundError(f"Critical file not found: {path}")

        # Load Plain text model
        with open(files["plain_model"], "rb") as f:
            model_p_data = pickle.load(f)
        vocab_p = 15000
        if files["plain_meta"].exists():
            with open(files["plain_meta"]) as f:
                vocab_p = json.load(f).get("vocab_size", 15000)

        self.model_plain = ArticleLinkerYOLO(vocab_size=vocab_p)
        self.model_plain.model = model_p_data

        with open(files["plain_vec"], "rb") as f:
            self.vec_plain = pickle.load(f)

        # Load Title model
        with open(files["title_model"], "rb") as f:
            model_t_data = pickle.load(f)
        vocab_t = 15000
        if files["title_meta"].exists():
            with open(files["title_meta"]) as f:
                vocab_t = json.load(f).get("vocab_size", 15000)

        self.model_titles = ArticleLinkerTitlesYOLO(vocab_size=vocab_t)
        self.model_titles.model = model_t_data

        with open(files["title_vec"], "rb") as f:
            self.vec_titles = pickle.load(f)

        # Load YOLO Embedder
        weights_path = MODEL_DIR / yolo_weights_name
        if not weights_path.exists():
            weights_path = Path(yolo_weights_name)

        print(f"Initializing YOLO Embedder ({weights_path.name})...")
        self.yolo_embedder = YoloEmbedder.get_instance(str(weights_path))
        if self.yolo_embedder._model_wrapper is None:
            print("Warning: YOLO detector not loaded. Using only text/geometric features.")
        else:
            print("Models loaded successfully.")

    @staticmethod
    def _safe_save_json(data, filepath):
        """Saves data to a JSON file atomically using a temporary file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(filepath) or ".",
            delete=False,
            suffix=".tmp",
        ) as tmp_file:
            json.dump(data, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            temp_name = tmp_file.name
        os.replace(temp_name, filepath)

    def _assemble_page(self, page_data):
        """
        Assembles text blocks on a single page into articles.

        The image path is taken exclusively from page_data['img_path'].

        Returns:
            tuple: (list of articles, dict mapping block index to article ID)
        """
        img_path_str = page_data.get("img_path")
        labels = page_data.get("labels", [])
        blocks = []

        # Collect valid blocks
        for label in labels:
            box_type = label.get("box_type", "plain text")
            if box_type not in ["plain text", "title"]:
                continue

            text = label.get("text_corrected", "")
            if not text or not text.strip():
                continue

            coords = label.get("box_coord", {})
            norm_coords = (
                float(coords.get("x1", 0)),
                float(coords.get("y1", 0)),
                float(coords.get("x2", 0)),
                float(coords.get("y2", 0)),
            )

            blocks.append(
                {
                    "index": label["index"],
                    "text": text,
                    "coords": norm_coords,
                    "box_type": box_type,
                    "_label_ref": label,
                }
            )

        n = len(blocks)
        if n == 0:
            return [], {}
        if n == 1:
            blocks[0]["article_id"] = 0
            blocks[0]["_label_ref"]["article_id"] = 0
            return [[blocks[0]]], {blocks[0]["index"]: 0}

        # Vectorize texts
        texts = [preprocess_text(b["text"]) for b in blocks]
        vectors_plain = list(self.vec_plain.transform(texts))
        vectors_titles = list(self.vec_titles.transform(texts))

        # Check if image exists and YOLO is available
        use_yolo = False
        if img_path_str:
            img_path_obj = Path(img_path_str)
            if (
                img_path_obj.exists()
                and self.yolo_embedder
                and self.yolo_embedder._model_wrapper is not None
            ):
                use_yolo = True

        # Build distance matrix
        distance_matrix = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(distance_matrix, 0.0)

        pairs = list(combinations(range(n), 2))
        total_pairs = len(pairs)

        if total_pairs > 0:
            print(f"Processing {total_pairs} block pairs...")

        for i, j in tqdm(pairs, total=total_pairs, desc="Pairs", unit="pair"):
            type_i, type_j = blocks[i]["box_type"], blocks[j]["box_type"]
            c1, c2 = blocks[i]["coords"], blocks[j]["coords"]

            yolo_emb = None
            if use_yolo:
                yolo_emb = self.yolo_embedder.get_embedding(img_path_str, c1, c2)

            if type_i == "plain text" and type_j == "plain text":
                v1, v2 = vectors_plain[i], vectors_plain[j]
                prob = self.model_plain.predict(v1, c1, v2, c2, yolo_emb=yolo_emb)
            else:
                v1, v2 = vectors_titles[i], vectors_titles[j]
                prob = self.model_titles.predict(
                    v1, c1, v2, c2, yolo_emb=yolo_emb, type1=type_i, type2=type_j
                )

            dist = 1.0 - float(prob)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

        # Clustering
        clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=(1.0 - self.threshold),
            n_clusters=None,
        )

        try:
            cluster_labels = clustering.fit_predict(distance_matrix)
        except Exception as e:
            print(f"Clustering error: {e}")
            return [], {}

        # Group blocks and assign article IDs
        articles_map = {}
        index_to_id = {}

        for idx, cluster_id in enumerate(cluster_labels):
            block = blocks[idx]
            cid = int(cluster_id)
            block["article_id"] = cid
            block["_label_ref"]["article_id"] = cid

            index_to_id[block["index"]] = cid
            articles_map.setdefault(cid, []).append(block)

        # Sort articles top-to-bottom
        sorted_articles = sorted(
            articles_map.values(), key=lambda art: min(b["coords"][1] for b in art)
        )

        # Reassign sequential IDs
        final_articles = []
        for new_id, article in enumerate(sorted_articles):
            for block in article:
                block["article_id"] = new_id
                block["_label_ref"]["article_id"] = new_id
                index_to_id[block["index"]] = new_id

            article.sort(key=lambda b: (b["coords"][1], b["coords"][0]))
            final_articles.append(article)

        return final_articles, index_to_id

    @staticmethod
    def _visualize_page(articles, img_path_full, output_path, max_width=1200):
        """Generates a visualization of assembled articles on the page image."""
        if not articles or not img_path_full or not Path(img_path_full).exists():
            return

        img = Image.open(img_path_full).convert("RGB")
        orig_w, orig_h = img.size

        scale = 1.0
        if orig_w > max_width:
            scale = max_width / orig_w
            img = img.resize(
                (int(orig_w * scale), int(orig_h * scale)), Image.Resampling.LANCZOS
            )

        draw = ImageDraw.Draw(img)
        palette = [
            "#FF5733",
            "#33FF57",
            "#3357FF",
            "#F333FF",
            "#FF33A8",
            "#33FFF5",
            "#F5FF33",
            "#FF8C33",
        ]
        colors = {i: palette[i % len(palette)] for i in range(len(articles))}

        font_size = max(10, int(img.width / 60))
        font = ImageFont.load_default()
        for path in ["Arial.ttf", "DejaVuSans.ttf"]:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except OSError:
                continue

        article_font_size = max(14, int(img.width / 40))
        article_font = None
        for path in ["Arial.ttf", "DejaVuSans.ttf"]:
            try:
                article_font = ImageFont.truetype(path, article_font_size)
                break
            except OSError:
                continue
        if article_font is None:
            article_font = font

        for art_id, blocks in enumerate(articles):
            color = colors[art_id]
            rgb = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
            article_num = art_id + 1

            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            o_draw = ImageDraw.Draw(overlay)

            for b in blocks:
                x1 = int(b["coords"][0] * orig_w * scale)
                y1 = int(b["coords"][1] * orig_h * scale)
                x2 = int(b["coords"][2] * orig_w * scale)
                y2 = int(b["coords"][3] * orig_h * scale)
                o_draw.rectangle([x1, y1, x2, y2], fill=(*rgb, 40))

            img = Image.alpha_composite(img.convert("RGBA"), overlay)
            draw = ImageDraw.Draw(img)

            for b in blocks:
                x1 = int(b["coords"][0] * orig_w * scale)
                y1 = int(b["coords"][1] * orig_h * scale)
                x2 = int(b["coords"][2] * orig_w * scale)
                y2 = int(b["coords"][3] * orig_h * scale)

                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                b_type = b.get("box_type", "plain text")
                t_short = "T" if b_type == "title" else "P"
                block_lbl = f"{t_short}#{b['index']}"

                bbox = draw.textbbox((x1, y1 - font_size - 2), block_lbl, font=font)
                draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill="black")
                draw.text((x1, y1 - font_size - 2), block_lbl, fill="white", font=font)

                num_text = str(article_num)
                num_bbox = draw.textbbox((0, 0), num_text, font=article_font)
                num_w = num_bbox[2] - num_bbox[0]
                num_h = num_bbox[3] - num_bbox[1]

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                pos_x = cx - num_w // 2
                pos_y = cy - num_h // 2

                padding = 4
                bg_box = [
                    pos_x - padding,
                    pos_y - padding,
                    pos_x + num_w + padding,
                    pos_y + num_h + padding,
                ]
                draw.rectangle(bg_box, fill=(*rgb, 200))
                draw.rectangle(bg_box, outline="white", width=1)
                draw.text((pos_x, pos_y), num_text, fill="white", font=article_font)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        img.save(output_path, optimize=True, quality=85)

    def process_file(self, input_json_path, output_json_path, viz_output_dir=None):
        """
        Main processing method.

        Image paths are taken directly from the JSON file.

        Args:
            input_json_path: Path to input JSON file.
            output_json_path: Path to save the output JSON.
            viz_output_dir: Optional directory for visualization outputs.

        Returns:
            list: The processed dataset with article assignments.
        """
        print(f"Reading file: {input_json_path}")
        with open(input_json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            raise ValueError("Input JSON must be a list of pages.")

        total_pages = len(dataset)
        print(f"Processing {total_pages} pages...")

        for idx, page in enumerate(dataset):
            img_path_str = page.get("img_path")
            img_path_obj = Path(img_path_str) if img_path_str else None

            articles, index_map = self._assemble_page(page)

            if viz_output_dir and img_path_obj and img_path_obj.exists() and articles:
                os.makedirs(viz_output_dir, exist_ok=True)
                fname = img_path_obj.stem
                viz_path = Path(viz_output_dir) / f"{fname}_articles.png"
                self._visualize_page(articles, str(img_path_obj), viz_path)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{total_pages} pages")

        print(f"Saving result to: {output_json_path}")
        self._safe_save_json(dataset, output_json_path)
        print("Done!")
        return dataset


if __name__ == "__main__":
    assembler = ArticleAssembler(threshold=0.5)
    assembler.process_file(
        input_json_path="dataset_with_final_correction.json",
        output_json_path="dataset_with_articles.json",
        viz_output_dir="visualizations",
    )
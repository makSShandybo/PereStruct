#!/usr/bin/env python3
"""
Unified dataset creation for both Plain-Text and Title models.
- Saves/loads splits from splits.pkl to ensure consistency between models
- Generates pairs_soviet.pkl (plain text model) 
- Generates pairs_soviet_title.pkl (title model)
- Uses parallel processing with thread-safe YOLO embedders
"""

import json
import pickle
import random
import threading
import argparse
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from pathlib import Path

from config import PROCESSED_DIR, IMAGES_FOLDER
from perestruct_classifier_train.preprocess import preprocess_text, YoloEmbedder

# ================================
# CONFIGURATION
# ================================
INPUT_JSON = "data/dataset_reindexed.json"
YOLO_WEIGHTS_PATH = "soviet_yolo.pt"
YOLO_DEVICE_ID = 0
MAX_WORKERS = 8
PLAINTEXT_MAX_NEG_POOL = 50000
TITLE_MAX_NEG_POOL = 20000

# ================================
# DATA LOADING & SPLITS (with caching)
# ================================
def load_or_create_splits(json_path, splits_path, train_ratio=0.84, val_ratio=0.08, random_state=42):
    """
    Load splits from cache if exists, otherwise create and save.
    Ensures both models use identical train/val/test splits.
    """
    splits_path = Path(splits_path)
    
    if splits_path.exists():
        print(f"Loading cached splits from {splits_path}")
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
        print(f"Splits loaded: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        return splits
    
    print(f"Creating new splits from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Checking block indices...")
    for page_idx, page in enumerate(data):
        for i, block in enumerate(page.get('labels', [])):
            if i != block.get('index'):
                print(f'Warning: Page {page_idx}, block index mismatch')
    
    np.random.seed(random_state)
    n = len(data)
    indices = np.random.permutation(n)
    
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    
    splits = {
        'train': [data[i] for i in indices[:train_size]],
        'val': [data[i] for i in indices[train_size:train_size + val_size]],
        'test': [data[i] for i in indices[train_size + val_size:]]
    }
    
    # Save splits for consistency
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(splits_path, 'wb') as f:
        pickle.dump(splits, f)
    
    print(f"Splits created and saved: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits

# ================================
# GLOBAL NEGATIVE POOLS
# ================================
def build_global_pool_plain(dataset, max_pool_size, embedder):
    """Build global negative pool for plain text model (text-text pairs only)."""
    print(f"Building global negative pool for Plain model...")
    pool = []
    all_blocks_by_page = []
    allowed = {'plain text'}
    
    for page in tqdm(dataset, desc="Scanning pages (Plain)"):
        labels = page.get('labels', [])
        img_path = page.get('img_path')
        img_w = page.get('w', 0)
        img_h = page.get('h', 0)
        full_img_path = str(IMAGES_FOLDER / img_path) if img_path else None

        page_blocks = []
        for label in labels:
            if label.get('box_type') not in allowed:
                continue
            text_raw = label.get('text_corrected', '') or label.get('text', '')
            if not text_raw or not text_raw.strip():
                continue
            
            text_processed = preprocess_text(text_raw)
            if not text_processed:
                continue
                
            coords = label.get('box_coord', {})
            coords_norm = (coords.get('x1', 0), coords.get('y1', 0), 
                          coords.get('x2', 0), coords.get('y2', 0))
            
            page_blocks.append({
                'text_processed': text_processed,
                'coords_norm': coords_norm,
                'img_path': full_img_path,
                'img_size': (img_w, img_h),
                'box_type': 'plain text'
            })
        
        if page_blocks:
            all_blocks_by_page.append(page_blocks)
    
    n_pages = len(all_blocks_by_page)
    if n_pages < 2:
        print("Warning: Too few pages for inter-page negatives")
        return []

    target_count = max_pool_size // 2
    attempts = 0
    max_attempts = target_count * 10
    
    with tqdm(total=target_count, desc="Generating Plain pool", unit="pairs") as pbar:
        while len(pool) < target_count and attempts < max_attempts:
            attempts += 1
            idx1, idx2 = random.sample(range(n_pages), 2)
            blocks1, blocks2 = all_blocks_by_page[idx1], all_blocks_by_page[idx2]
            
            if not blocks1 or not blocks2:
                continue
                
            b1 = random.choice(blocks1)
            b2 = random.choice(blocks2)
            
            yolo_emb = np.zeros(576, dtype=np.float32)
            if embedder:
                yolo_emb = embedder.get_embedding(b1['img_path'], b1['coords_norm'], b2['coords_norm'])

            pair_1 = {
                'text1': b1['text_processed'], 'text2': b2['text_processed'],
                'c1': b1['coords_norm'], 'c2': b2['coords_norm'],
                'img_path': b1['img_path'], 'img_size': b1['img_size'],
                'label': 0, 'type1': 'plain text', 'type2': 'plain text',
                'pair_type': 'plain text ↔ plain text (inter-page)',
                'is_inter_page': True, 'yolo_emb': yolo_emb
            }
            pair_2 = {
                'text1': b2['text_processed'], 'text2': b1['text_processed'],
                'c1': b2['coords_norm'], 'c2': b1['coords_norm'],
                'img_path': b2['img_path'], 'img_size': b2['img_size'],
                'label': 0, 'type1': 'plain text', 'type2': 'plain text',
                'pair_type': 'plain text ↔ plain text (inter-page)',
                'is_inter_page': True, 'yolo_emb': yolo_emb
            }
            
            pool.extend([pair_1, pair_2])
            pbar.update(2)
            
    print(f"Plain global pool ready: {len(pool)} pairs")
    return pool


def build_global_pool_title(dataset, max_pool_size, embedder):
    """Build global negative pool for title model (pairs with at least one title)."""
    print(f"Building global negative pool for Title model...")
    pool = []
    all_blocks_by_page = []
    allowed = {'plain text', 'title'}
    
    for page in tqdm(dataset, desc="Scanning pages (Title)"):
        labels = page.get('labels', [])
        img_path = page.get('img_path')
        img_w = page.get('w', 0)
        img_h = page.get('h', 0)
        full_img_path = str(IMAGES_FOLDER / img_path) if img_path else None

        page_blocks = []
        for label in labels:
            box_type = label.get('box_type', '')
            if box_type not in allowed:
                continue
            
            text_raw = label.get('text_corrected', '') or label.get('text', '')
            if not text_raw or not text_raw.strip():
                continue
            
            text_processed = preprocess_text(text_raw)
            if not text_processed:
                continue
                
            coords = label.get('box_coord', {})
            coords_norm = (coords.get('x1', 0), coords.get('y1', 0),
                          coords.get('x2', 0), coords.get('y2', 0))
            
            page_blocks.append({
                'text_processed': text_processed,
                'coords_norm': coords_norm,
                'img_path': full_img_path,
                'img_size': (img_w, img_h),
                'box_type': box_type
            })
        
        # Keep page only if it has at least one title
        has_title = any(b['box_type'] == 'title' for b in page_blocks)
        if has_title and page_blocks:
            all_blocks_by_page.append(page_blocks)
    
    n_pages = len(all_blocks_by_page)
    if n_pages < 2:
        print("Warning: Too few pages with titles")
        return []

    target_count = max_pool_size // 2
    attempts = 0
    max_attempts = target_count * 10
    
    with tqdm(total=target_count, desc="Generating Title pool", unit="pairs") as pbar:
        while len(pool) < target_count and attempts < max_attempts:
            attempts += 1
            idx1, idx2 = random.sample(range(n_pages), 2)
            blocks1, blocks2 = all_blocks_by_page[idx1], all_blocks_by_page[idx2]
            
            if not blocks1 or not blocks2:
                continue
                
            b1 = random.choice(blocks1)
            b2 = random.choice(blocks2)
            
            # Filter: at least one must be title
            if b1['box_type'] != 'title' and b2['box_type'] != 'title':
                continue
            
            yolo_emb = np.zeros(576, dtype=np.float32)
            if embedder:
                yolo_emb = embedder.get_embedding(b1['img_path'], b1['coords_norm'], b2['coords_norm'])

            pair_1 = {
                'text1': b1['text_processed'], 'text2': b2['text_processed'],
                'c1': b1['coords_norm'], 'c2': b2['coords_norm'],
                'img_path': b1['img_path'], 'img_size': b1['img_size'],
                'label': 0, 'type1': b1['box_type'], 'type2': b2['box_type'],
                'pair_type': f"{b1['box_type']} ↔ {b2['box_type']} (inter-page)",
                'is_inter_page': True, 'yolo_emb': yolo_emb
            }
            pair_2 = {
                'text1': b2['text_processed'], 'text2': b1['text_processed'],
                'c1': b2['coords_norm'], 'c2': b1['coords_norm'],
                'img_path': b2['img_path'], 'img_size': b2['img_size'],
                'label': 0, 'type1': b2['box_type'], 'type2': b1['box_type'],
                'pair_type': f"{b2['box_type']} ↔ {b1['box_type']} (inter-page)",
                'is_inter_page': True, 'yolo_emb': yolo_emb
            }
            
            pool.extend([pair_1, pair_2])
            pbar.update(2)
            
    print(f"Title global pool ready: {len(pool)} pairs")
    return pool

# ================================
# PAGE PROCESSING (Plain Model)
# ================================
def process_page_plain(args):
    """Process single page for plain text model."""
    page, page_idx, embedder, global_neg_pool = args
    img_path = page.get('img_path')
    labels = page.get('labels', [])
    articles = page.get('articles', {})
    
    if not img_path or not labels:
        return [], []
    
    full_img_path = str(IMAGES_FOLDER / img_path) if img_path else None
    img_w = page.get('w', 0)
    img_h = page.get('h', 0)
    
    block_to_article = {}
    if isinstance(articles, dict):
        for art_id, indices in articles.items():
            if isinstance(indices, list):
                for idx in indices:
                    block_to_article[idx] = art_id
            elif isinstance(indices, dict):
                for idx in indices.keys():
                    block_to_article[int(idx)] = art_id
    
    block_data = []
    for label in labels:
        if label.get('box_type') != 'plain text':
            continue
        text_raw = label.get('text_corrected', '') or label.get('text', '')
        if not text_raw or not text_raw.strip():
            continue
        
        text_processed = preprocess_text(text_raw)
        if not text_processed:
            continue
        
        coords = label.get('box_coord', {})
        coords_norm = (coords.get('x1', 0), coords.get('y1', 0),
                      coords.get('x2', 0), coords.get('y2', 0))
        
        article_id = label.get('article') or block_to_article.get(label['index'])
        
        block_data.append({
            'text_processed': text_processed,
            'coords_norm': coords_norm,
            'article': article_id,
            'box_type': 'plain text',
            'img_path': full_img_path,
            'img_size': (img_w, img_h),
            'index': label['index']
        })
    
    if len(block_data) < 2:
        return [], []
    
    art_blocks = {}
    for i, b in enumerate(block_data):
        if b['article'] is not None:
            art_blocks.setdefault(b['article'], []).append(i)
    
    if len(art_blocks) == 0:
        return [], []
    
    # Positive pairs
    pos_pairs = []
    for indices in art_blocks.values():
        if len(indices) < 2:
            continue
        for i, j in combinations(indices, 2):
            bi, bj = block_data[i], block_data[j]
            
            yolo_emb = np.zeros(576, dtype=np.float32)
            if embedder:
                yolo_emb = embedder.get_embedding(bi['img_path'], bi['coords_norm'], bj['coords_norm'])

            base = {
                'text1': bi['text_processed'], 'text2': bj['text_processed'],
                'c1': bi['coords_norm'], 'c2': bj['coords_norm'],
                'img_path': bi['img_path'], 'img_size': bi['img_size'],
                'label': 1, 'article_id': bi['article'],
                'type1': 'plain text', 'type2': 'plain text',
                'pair_type': "plain text ↔ plain text", 'yolo_emb': yolo_emb
            }
            pos_pairs.append(base)
            pos_pairs.append({**base, 'text1': bj['text_processed'], 'text2': bi['text_processed'],
                            'c1': bj['coords_norm'], 'c2': bi['coords_norm']})
    
    # Negative pairs
    neg_pairs = []
    arts = list(art_blocks.values())
    local_neg = []
    
    if len(arts) > 1:
        for i in range(len(arts)):
            for j in range(i + 1, len(arts)):
                for a_idx in arts[i]:
                    for b_idx in arts[j]:
                        ba, bb = block_data[a_idx], block_data[b_idx]
                        
                        yolo_emb = np.zeros(576, dtype=np.float32)
                        if embedder:
                            yolo_emb = embedder.get_embedding(ba['img_path'], ba['coords_norm'], bb['coords_norm'])

                        cand = {
                            'text1': ba['text_processed'], 'text2': bb['text_processed'],
                            'c1': ba['coords_norm'], 'c2': bb['coords_norm'],
                            'img_path': ba['img_path'], 'img_size': ba['img_size'],
                            'label': 0, 'type1': 'plain text', 'type2': 'plain text',
                            'is_inter_page': False, 'yolo_emb': yolo_emb
                        }
                        local_neg.append(cand)
                        local_neg.append({**cand, 'text1': bb['text_processed'], 'text2': ba['text_processed'],
                                        'c1': bb['coords_norm'], 'c2': ba['coords_norm']})

    n_pos = len(pos_pairs)
    if n_pos == 0:
        return [], []
    
    random.seed(42 + page_idx)
    
    if len(local_neg) >= n_pos:
        neg_pairs = random.sample(local_neg, n_pos)
    else:
        neg_pairs = local_neg[:]
        needed = n_pos - len(neg_pairs)
        if global_neg_pool:
            extra = random.sample(global_neg_pool, min(needed, len(global_neg_pool)))
            if len(extra) < needed:
                repeat = (needed // len(global_neg_pool)) + 1
                extra = (global_neg_pool * repeat)[:needed]
            neg_pairs.extend(extra)
    
    return pos_pairs, neg_pairs


# ================================
# PAGE PROCESSING (Title Model)
# ================================
def process_page_title(args):
    """Process single page for title model."""
    page, page_idx, embedder, global_neg_pool = args
    img_path = page.get('img_path')
    labels = page.get('labels', [])
    articles = page.get('articles', {})
    
    if not img_path or not labels:
        return [], []
    
    full_img_path = str(IMAGES_FOLDER / img_path) if img_path else None
    img_w = page.get('w', 0)
    img_h = page.get('h', 0)
    
    block_to_article = {}
    if isinstance(articles, dict):
        for art_id, indices in articles.items():
            if isinstance(indices, list):
                for idx in indices:
                    block_to_article[idx] = art_id
            elif isinstance(indices, dict):
                for idx in indices.keys():
                    block_to_article[int(idx)] = art_id
    
    block_data = []
    for label in labels:
        box_type = label.get('box_type', '')
        if box_type not in {'plain text', 'title'}:
            continue
        
        text_raw = label.get('text_corrected', '') or label.get('text', '')
        if not text_raw or not text_raw.strip():
            continue
        
        text_processed = preprocess_text(text_raw)
        if not text_processed:
            continue
        
        coords = label.get('box_coord', {})
        coords_norm = (coords.get('x1', 0), coords.get('y1', 0),
                      coords.get('x2', 0), coords.get('y2', 0))
        
        article_id = label.get('article') or block_to_article.get(label['index'])
        
        block_data.append({
            'text_processed': text_processed,
            'coords_norm': coords_norm,
            'article': article_id,
            'box_type': box_type,
            'img_path': full_img_path,
            'img_size': (img_w, img_h),
            'index': label['index']
        })
    
    if len(block_data) < 2:
        return [], []
    
    art_blocks = {}
    orphan_titles = []
    
    for i, b in enumerate(block_data):
        if b['article'] is not None:
            art_blocks.setdefault(b['article'], []).append(i)
        elif b['box_type'] == 'title':
            orphan_titles.append(i)
    
    # Positive pairs (at least one title)
    pos_pairs = []
    for indices in art_blocks.values():
        if len(indices) < 2:
            continue
        for i, j in combinations(indices, 2):
            bi, bj = block_data[i], block_data[j]
            
            if bi['box_type'] != 'title' and bj['box_type'] != 'title':
                continue
            
            yolo_emb = np.zeros(576, dtype=np.float32)
            if embedder:
                yolo_emb = embedder.get_embedding(bi['img_path'], bi['coords_norm'], bj['coords_norm'])

            base = {
                'text1': bi['text_processed'], 'text2': bj['text_processed'],
                'c1': bi['coords_norm'], 'c2': bj['coords_norm'],
                'img_path': bi['img_path'], 'img_size': bi['img_size'],
                'label': 1, 'article_id': bi['article'],
                'type1': bi['box_type'], 'type2': bj['box_type'],
                'pair_type': f"{bi['box_type']} ↔ {bj['box_type']}", 'yolo_emb': yolo_emb
            }
            pos_pairs.append(base)
            pos_pairs.append({**base, 'text1': bj['text_processed'], 'text2': bi['text_processed'],
                            'c1': bj['coords_norm'], 'c2': bi['coords_norm']})
    
    # Negative pairs (at least one title)
    neg_pairs = []
    arts = list(art_blocks.keys())
    local_neg = []
    
    if len(arts) > 1:
        for i_idx in range(len(arts)):
            for j_idx in range(i_idx + 1, len(arts)):
                for a_idx in art_blocks[arts[i_idx]]:
                    for b_idx in art_blocks[arts[j_idx]]:
                        ba, bb = block_data[a_idx], block_data[b_idx]
                        
                        if ba['box_type'] != 'title' and bb['box_type'] != 'title':
                            continue

                        yolo_emb = np.zeros(576, dtype=np.float32)
                        if embedder:
                            yolo_emb = embedder.get_embedding(ba['img_path'], ba['coords_norm'], bb['coords_norm'])

                        cand = {
                            'text1': ba['text_processed'], 'text2': bb['text_processed'],
                            'c1': ba['coords_norm'], 'c2': bb['coords_norm'],
                            'img_path': ba['img_path'], 'img_size': ba['img_size'],
                            'label': 0, 'type1': ba['box_type'], 'type2': bb['box_type'],
                            'is_inter_page': False, 'yolo_emb': yolo_emb
                        }
                        local_neg.append(cand)
                        local_neg.append({**cand, 'text1': bb['text_processed'], 'text2': ba['text_processed'],
                                        'c1': bb['coords_norm'], 'c2': ba['coords_norm']})
    
    # Orphan titles as negatives
    if orphan_titles:
        other_blocks = [i for i, b in enumerate(block_data) if b['article'] is not None]
        for t_idx in orphan_titles:
            title_block = block_data[t_idx]
            targets = random.sample(other_blocks, min(5, len(other_blocks))) if other_blocks else []
            for o_idx in targets:
                other_block = block_data[o_idx]
                
                yolo_emb = np.zeros(576, dtype=np.float32)
                if embedder:
                    yolo_emb = embedder.get_embedding(title_block['img_path'], 
                                                     title_block['coords_norm'], 
                                                     other_block['coords_norm'])
                
                local_neg.append({
                    'text1': title_block['text_processed'], 'text2': other_block['text_processed'],
                    'c1': title_block['coords_norm'], 'c2': other_block['coords_norm'],
                    'img_path': title_block['img_path'], 'img_size': title_block['img_size'],
                    'label': 0, 'type1': 'title', 'type2': other_block['box_type'],
                    'pair_type': "title (orphan) ↔ text", 'is_inter_page': False, 'yolo_emb': yolo_emb
                })

    n_pos = len(pos_pairs)
    if n_pos == 0:
        return [], []
    
    random.seed(42 + page_idx)
    
    if len(local_neg) >= n_pos:
        neg_pairs = random.sample(local_neg, n_pos)
    else:
        neg_pairs = local_neg[:]
        needed = n_pos - len(neg_pairs)
        if global_neg_pool:
            extra = random.sample(global_neg_pool, min(needed, len(global_neg_pool)))
            if len(extra) < needed:
                repeat = (needed // len(global_neg_pool)) + 1
                extra = (global_neg_pool * repeat)[:needed]
            neg_pairs.extend(extra)
    
    return pos_pairs, neg_pairs

# ================================
# PARALLEL EXECUTION
# ================================
def create_pairs_parallel(dataset, name, process_func, global_neg_pool, 
                         num_workers, device_id, weights_path):
    """Generic parallel processing for pair generation."""
    print(f"\nProcessing {name}: {num_workers} workers (CUDA:{device_id})...")
    
    thread_local = threading.local()

    def init_worker(weights, dev_id):
        le = YoloEmbedder()
        le._device = f'cuda:{dev_id}'
        le._model_wrapper = None
        le._img_cache = {}
        le.load_model(weights)
        thread_local.embedder = le

    def run_task(task_args):
        page, idx, _, g_pool = task_args
        local_embedder = getattr(thread_local, 'embedder', None)
        return process_func((page, idx, local_embedder, g_pool))

    all_pos = []
    all_neg = []
    
    with ThreadPoolExecutor(max_workers=num_workers, initializer=init_worker, 
                           initargs=(weights_path, device_id)) as executor:
        futures = [executor.submit(run_task, (page, idx, None, global_neg_pool)) 
                   for idx, page in enumerate(dataset)]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {name}"):
            pos, neg = future.result()
            all_pos.extend(pos)
            all_neg.extend(neg)
            
    return all_pos, all_neg

# ================================
# MAIN
# ================================
def main():
    parser = argparse.ArgumentParser(description="Generate both Plain and Title model datasets")
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--device', type=int, default=YOLO_DEVICE_ID)
    parser.add_argument('--weights', type=str, default=YOLO_WEIGHTS_PATH)
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    splits_path = PROCESSED_DIR / "splits.pkl"
    
    print("=" * 70)
    print("DATASET GENERATION - BOTH MODELS")
    print("=" * 70)
    
    splits = load_or_create_splits(INPUT_JSON, splits_path)
    
    # ================================
    # PLAIN MODEL
    # ================================
    print("\n" + "=" * 70)
    print("PLAIN TEXT MODEL")
    print("=" * 70)
    
    print("\nInitializing embedder for Plain global pool...")
    pool_embedder_plain = YoloEmbedder()
    pool_embedder_plain._device = f'cuda:{args.device}'
    pool_embedder_plain.load_model(args.weights)
    
    global_pool_plain = build_global_pool_plain(splits['train'], PLAINTEXT_MAX_NEG_POOL, pool_embedder_plain)
    del pool_embedder_plain
    
    train_plain = create_pairs_parallel(splits['train'], "Train (Plain)", process_page_plain, 
                                       global_pool_plain, args.workers, args.device, args.weights)
    val_plain = create_pairs_parallel(splits['val'], "Val (Plain)", process_page_plain, 
                                     global_pool_plain, args.workers, args.device, args.weights)
    test_plain = create_pairs_parallel(splits['test'], "Test (Plain)", process_page_plain, 
                                      global_pool_plain, args.workers, args.device, args.weights)
    
    pairs_plain = {
        'train_pos': train_plain[0], 'train_neg': train_plain[1],
        'val_pos': val_plain[0], 'val_neg': val_plain[1],
        'test_pos': test_plain[0], 'test_neg': test_plain[1]
    }
    
    with open(PROCESSED_DIR / 'pairs_soviet.pkl', 'wb') as f:
        pickle.dump(pairs_plain, f)
    
    print(f"\nPlain model saved: pairs_soviet.pkl")
    print(f"  Train: {len(train_plain[0])} pos + {len(train_plain[1])} neg")
    print(f"  Val:   {len(val_plain[0])} pos + {len(val_plain[1])} neg")
    print(f"  Test:  {len(test_plain[0])} pos + {len(test_plain[1])} neg")
    
    # ================================
    # TITLE MODEL
    # ================================
    print("\n" + "=" * 70)
    print("TITLE MODEL")
    print("=" * 70)
    
    print("\nInitializing embedder for Title global pool...")
    pool_embedder_title = YoloEmbedder()
    pool_embedder_title._device = f'cuda:{args.device}'
    pool_embedder_title.load_model(args.weights)
    
    global_pool_title = build_global_pool_title(splits['train'], TITLE_MAX_NEG_POOL, pool_embedder_title)
    del pool_embedder_title
    
    train_title = create_pairs_parallel(splits['train'], "Train (Title)", process_page_title, 
                                         global_pool_title, args.workers, args.device, args.weights)
    val_title = create_pairs_parallel(splits['val'], "Val (Title)", process_page_title, 
                                       global_pool_title, args.workers, args.device, args.weights)
    test_title = create_pairs_parallel(splits['test'], "Test (Title)", process_page_title, 
                                        global_pool_title, args.workers, args.device, args.weights)
    
    pairs_title = {
        'train_pos': train_title[0], 'train_neg': train_title[1],
        'val_pos': val_title[0], 'val_neg': val_title[1],
        'test_pos': test_title[0], 'test_neg': test_title[1]
    }
    
    with open(PROCESSED_DIR / 'pairs_soviet_title.pkl', 'wb') as f:
        pickle.dump(pairs_title, f)
    
    print(f"\nTitle model saved: pairs_soviet_title.pkl")
    print(f"  Train: {len(train_title[0])} pos + {len(train_title[1])} neg")
    print(f"  Val:   {len(val_title[0])} pos + {len(val_title[1])} neg")
    print(f"  Test:  {len(test_title[0])} pos + {len(test_title[1])} neg")
    
    # Final summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Splits cached at: {splits_path}")
    print(f"Plain pairs:  pairs_soviet.pkl ({sum(len(pairs_plain[k]) for k in pairs_plain)} total)")
    print(f"Title pairs:  pairs_soviet_title.pkl ({sum(len(pairs_title[k]) for k in pairs_title)} total)")
    print("Done!")

if __name__ == '__main__':
    main()

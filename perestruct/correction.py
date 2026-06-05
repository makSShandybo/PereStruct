import json
import os
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm

from yandex_cloud_ml_sdk import YCloudML

from .settings import settings

API_KEY = settings.llm_api_key
FOLDER_ID = settings.folder_id

DEFAULT_INPUT_JSON = "final_data/dataset_with_final_correction.json"
DEFAULT_OUTPUT_JSON = "final_data/dataset_with_final_correction.json"
DEFAULT_MAX_WORKERS = 8
DEFAULT_OCR_TYPES = {"title", "plain text"}

WRITE_LOCK = Lock()

CORRECTION_PROMPT = """
Ты — эксперт по русскому языку и историческим текстам, особенно по материалам советской эпохи. Твоя задача — исправить только явные опечатки, возникшие в результате ошибок OCR (распознавания текста с изображений), не внося никаких других изменений в текст.

Правила:

Не перефразируй, не улучшай стиль, не «современизируй» лексику и не исправляй грамматические конструкции, характерные для советской эпохи.
Сохраняй оригинальную пунктуацию, регистр букв, орфографию и стилистику, даже если они кажутся устаревшими или не соответствуют современным нормам.
Исправляй только те символы или слова, которые очевидно искажены OCR (например: «сегоднл» → «сегодня», «ро6очий» → «рабочий», «пoддержкa» → «поддержка»).
Если сомневаешься — оставляй как есть.
Не добавляй, не удаляй и не переставляй слова, предложения или абзацы.
Ты должен удалить \\n и исправить переносы слов (пример: при-/n своенный -> присвоенный).
Никогда не меняй регистр букв (с маленькой на заглавуную и наоборот).
Верни только исправленный текст, без пояснений, комментариев или форматирования. 
"""


def safe_save_json(data, filepath):
    """Saves data to a JSON file atomically using a temporary file."""
    dir_path = os.path.dirname(filepath) or '.'
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=dir_path,
        delete=False, suffix='.tmp'
    ) as tmp_file:
        json.dump(data, tmp_file, ensure_ascii=False, indent=2)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_name = tmp_file.name
    os.replace(temp_name, filepath)


def correct_single_block(text, model, max_retries=2):
    """
    Sends text to the model for correction.

    Returns:
        str: Corrected text on success.
        str: Empty string if input is empty.
        None: On API error (requires retry).
    """
    if not text or not text.strip():
        return ""

    for attempt in range(max_retries):
        try:
            message = [
                {"role": "system", "text": CORRECTION_PROMPT},
                {"role": "user", "text": text}
            ]

            operation = model.configure(temperature=0.5).run_deferred(message)
            answer = operation.wait(timeout=60)
            corrected_text = answer[0].text.replace('\n', ' ')

            return corrected_text if corrected_text else ""

        except Exception as e:
            error_str = str(e)

            if "PERMISSION_DENIED" in error_str or "AUTH" in error_str.upper():
                print(f" Critical error: {error_str[:80]}")
                return None

            if attempt < max_retries - 1:
                print(f" Attempt {attempt + 1}/{max_retries} failed, retrying...")
                time.sleep(2)
            else:
                print(f" Error after {max_retries} attempts: {error_str[:80]}")
                return None

    return None


def analyze_coverage(dataset, ocr_types=None):
    """Analyzes the coverage and status of OCR blocks in the dataset."""
    if ocr_types is None:
        ocr_types = {"title", "plain text"}

    stats = defaultdict(lambda: {
        "total": 0,
        "has_ocr_res": 0,
        "has_text": 0,
        "has_text_corrected_nonempty": 0,
        "has_text_corrected_empty": 0,
        "text_changed": 0,
    })

    for page_idx, page in enumerate(dataset):
        for box in page.get("labels", []):
            box_type = box.get("box_type", "unknown")

            if box_type not in ocr_types:
                continue

            s = stats[box_type]
            s["total"] += 1

            if box.get("ocr_res_ful"):
                s["has_ocr_res"] += 1

            if box.get("text"):
                s["has_text"] += 1

            tc = box.get("text_corrected")
            if tc is not None:
                if tc.strip():
                    s["has_text_corrected_nonempty"] += 1
                    if tc != box.get("text", ""):
                        s["text_changed"] += 1
                else:
                    s["has_text_corrected_empty"] += 1

    return dict(stats)


def print_coverage_report(stats):
    """Prints a detailed report of block coverage statistics."""
    print("\n" + "=" * 80)
    print("BLOCK COVERAGE STATISTICS")
    print("=" * 80)

    for box_type, s in stats.items():
        total = s["total"]
        if total == 0:
            continue

        print(f"\n Type: {box_type.upper()} ({total} blocks)")
        print("-" * 60)

        ocr_pct = s["has_ocr_res"] / total * 100
        print(f"   Has ocr_res_ful:      {s['has_ocr_res']:5d} / {total:5d}  ({ocr_pct:5.1f}%)")

        text_pct = s["has_text"] / total * 100
        print(f"   Has text:             {s['has_text']:5d} / {total:5d}  ({text_pct:5.1f}%)")

        tc_ne_pct = s["has_text_corrected_nonempty"] / total * 100
        print(f"   text_corrected (non-empty): {s['has_text_corrected_nonempty']:5d} / {total:5d}  ({tc_ne_pct:5.1f}%)")

        tc_e_pct = s["has_text_corrected_empty"] / total * 100
        print(f"   text_corrected (empty):  {s['has_text_corrected_empty']:5d} / {total:5d}  ({tc_e_pct:5.1f}%)")

        if s["has_text_corrected_nonempty"] > 0:
            changed_pct = s["text_changed"] / s["has_text_corrected_nonempty"] * 100
            print(f"   Text Changed:         {s['text_changed']:5d} / {s['has_text_corrected_nonempty']:5d}  ({changed_pct:5.1f}%)")

        need_retry = s["has_text_corrected_empty"]
        if need_retry > 0:
            print(f"\n   Requires reprocessing:  {need_retry} blocks")

    print("\n" + "=" * 80)
    total_all = sum(s["total"] for s in stats.values())
    total_retry = sum(s["has_text_corrected_empty"] for s in stats.values())
    total_corrected = sum(s["has_text_corrected_nonempty"] for s in stats.values())

    if total_all > 0:
        print(f" TOTAL SUMMARY:")
        print(f"   Total OCR Blocks:      {total_all}")
        print(f"   Successfully Corrected: {total_corrected} ({total_corrected/total_all*100:.1f}%)")
        print(f"   Requires Reprocessing:   {total_retry} ({total_retry/total_all*100:.1f}%)")
    print("=" * 80)


def show_changed_examples(dataset, ocr_types=None, max_show=10):
    """Displays examples of blocks where text was changed after correction."""
    if ocr_types is None:
        ocr_types = {"title", "plain text"}

    changed = []
    for page_idx, page in enumerate(dataset):
        for box in page.get("labels", []):
            if box.get("box_type") not in ocr_types:
                continue

            original = box.get("text", "")
            corrected = box.get("text_corrected", "")

            if original and corrected and original != corrected:
                changed.append({
                    "page": page_idx,
                    "index": box.get("index"),
                    "box_type": box.get("box_type"),
                    "original": original,
                    "corrected": corrected
                })

    if not changed:
        print("\n No blocks with changes after correction")
        return

    print(f"\n Change Examples (showing {min(max_show, len(changed))} of {len(changed)}):")
    print("-" * 80)

    for i, item in enumerate(changed[:max_show], 1):
        print(f"\n[{i}]  Page {item['page']} | #{item['index']} | {item['box_type']}")
        orig_text = item['original']
        corr_text = item['corrected']
        print(f"   Before:  {orig_text[:100]}{'...' if len(orig_text) > 100 else ''}")
        print(f"   After: {corr_text[:100]}{'...' if len(corr_text) > 100 else ''}")

    if len(changed) > max_show:
        print(f"\n   ... and {len(changed) - max_show} more blocks")


def run_correction(
    input_path=DEFAULT_INPUT_JSON,
    output_path=DEFAULT_OUTPUT_JSON,
    max_workers=DEFAULT_MAX_WORKERS,
    ocr_types=DEFAULT_OCR_TYPES,
    show_report=True,
    show_examples=True
):
    """
    Runs the text correction process for blocks with empty text_corrected.

    Args:
        input_path: Path to the input JSON.
        output_path: Path to the output JSON (defaults to same as input).
        max_workers: Number of threads.
        ocr_types: Set of block types to process.
        show_report: Whether to print statistics.
        show_examples: Whether to print change examples.

    Returns:
        list: The processed dataset.
    """
    print("=" * 80)
    print(" STARTING CORRECTION RETRY")
    print("=" * 80)

    sdk = YCloudML(folder_id=FOLDER_ID, auth=API_KEY)
    model = sdk.models.completions("yandexgpt", model_version="rc")

    print(" Loading dataset...")
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if show_report:
        print("\n CURRENT STATE (before retry):")
        stats_before = analyze_coverage(dataset, ocr_types)
        print_coverage_report(stats_before)

    tasks = []
    for page in dataset:
        for box in page.get("labels", []):
            if box.get("box_type") not in ocr_types:
                continue

            tc = box.get("text_corrected")
            original_text = box.get("text", "")

            needs_processing = False
            if tc is None and original_text.strip():
                needs_processing = True
            elif tc is not None and not tc.strip() and original_text.strip():
                needs_processing = True

            if needs_processing:
                tasks.append((box, original_text))

    if not tasks:
        print("\n All blocks already processed! No retry needed.")
        if show_examples:
            show_changed_examples(dataset, ocr_types)
        return dataset

    print(f"\n STARTING RETRY FOR {len(tasks)} BLOCKS...")
    print(f" Parameters: {max_workers} threads")
    print("-" * 70)

    success = 0
    failed = 0

    def process_task(box, original_text):
        corrected = correct_single_block(original_text, model)

        with WRITE_LOCK:
            if corrected is not None:
                box["text_corrected"] = corrected
                safe_save_json(dataset, output_path)
                return True
            else:
                return False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_box = {
            executor.submit(process_task, box, text): box
            for (box, text) in tasks
        }

        for future in tqdm(as_completed(future_to_box), total=len(future_to_box), desc="Retrying"):
            try:
                if future.result():
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"❌ Error: {e}")

    print("-" * 70)
    print(f" Success: {success}")
    print(f" Errors (remaining for retry): {failed}")

    if show_report:
        print("\n STATE AFTER RETRY:")
        stats_after = analyze_coverage(dataset, ocr_types)
        print_coverage_report(stats_after)

        if show_examples:
            show_changed_examples(dataset, ocr_types)

    print(f"\n File saved: {output_path}")
    print("=" * 80)

    return dataset
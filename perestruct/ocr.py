import base64
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from threading import Lock

import requests
from PIL import Image
from tqdm import tqdm

from .settings import settings


API_KEY = settings.ocr_api_key
OUTPUT_JSON_PATH = "dataset_with_ocr.json"
OCR_TYPES = {"title", "plain text"}
MAX_WORKERS = 4

write_lock = Lock()


def safe_save_json(data, filepath):
    """Atomically save JSON to file."""
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=dir_name if dir_name else '.',
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        json.dump(data, tmp_file, ensure_ascii=False, indent=2)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_name = tmp_file.name

    os.replace(temp_name, filepath)


def call_api(url, data=None, method='POST'):
    """API request wrapper."""
    headers = {"Authorization": f"Api-Key {API_KEY}"}
    url = url.strip()

    try:
        if method == 'POST':
            resp = requests.post(url, json=data, headers=headers, timeout=30)
        elif method == 'GET':
            resp = requests.get(url, headers=headers, timeout=30)
        else:
            raise ValueError("Method must be POST or GET")

        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {e}")


def ocr_async(img_crop):
    """Send image to async OCR and return operation ID."""
    buffer = BytesIO()
    img_crop.save(buffer, format="JPEG", quality=100)
    image_bytes = buffer.getvalue()

    payload = {
        "mimeType": "image/jpeg",
        "languageCodes": ["ru"],
        "model": "page",
        "content": base64.b64encode(image_bytes).decode('utf-8')
    }

    res = call_api(
        "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeTextAsync",
        payload
    )

    if "id" not in res:
        raise Exception(f"Failed to create operation: {res}")
    return res["id"]


def wait_and_get_ocr_result(operation_id):
    """Wait for operation completion and return text."""
    status_url = (
        f"https://operation.api.cloud.yandex.net/operations/{operation_id}"
    )

    for _ in range(40):
        try:
            status = call_api(status_url, method='GET')
            if status.get("done"):
                if "error" in status:
                    raise Exception(f"OCR server error: {status['error']}")
                break
        except Exception as e:
            print(f"Status check error for {operation_id}: {e}")
        time.sleep(2)
    else:
        raise Exception(f"Timeout waiting for operation {operation_id}")

    result_url = (
        f"https://ocr.api.cloud.yandex.net/ocr/v1/getRecognition?"
        f"operationId={operation_id}"
    )
    result = call_api(result_url, method='GET')

    if "error" in result:
        raise Exception(f"Result retrieval error: {result['error']}")

    full_text = (
        result.get("result", {})
        .get("textAnnotation", {})
        .get("fullText", "")
    )
    return {"raw_recognition": result, "text": full_text}


def process_box_task(box, img_path_str, dataset_ref, output_path):
    """Process single box: crop, send to OCR, update dataset, save file."""
    if box.get("text") is not None:
        return

    try:
        with Image.open(img_path_str) as img:
            img_w, img_h = img.size

            x1 = int(box["box_coord"]["x1"] * img_w)
            y1 = int(box["box_coord"]["y1"] * img_h)
            x2 = int(box["box_coord"]["x2"] * img_w)
            y2 = int(box["box_coord"]["y2"] * img_h)

            cropped = img.crop((x1, y1, x2, y2))

            if cropped.width == 0 or cropped.height == 0:
                ocr_result = {"text": ""}
            else:
                op_id = ocr_async(cropped)
                ocr_result = wait_and_get_ocr_result(op_id)

        with write_lock:
            box["text"] = ocr_result["text"]
            safe_save_json(dataset_ref, output_path)

    except Exception as e:
        print(f"Error processing box ({img_path_str}): {e}")
        with write_lock:
            box["text"] = ""
            safe_save_json(dataset_ref, output_path)


def run_ocr(yolo_boxes_data, output_file=OUTPUT_JSON_PATH):
    """Process YOLO detection results and run OCR on text blocks."""
    print(f"Starting processing of {len(yolo_boxes_data)} images...")

    dataset = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            print(
                f"Found progress file '{output_file}'. "
                f"Loaded {len(dataset)} pages."
            )
        except json.JSONDecodeError:
            print("Progress file corrupted. Creating new one.")
            dataset = []

    if not dataset:
        dataset = yolo_boxes_data

        for page in dataset:
            for box in page["labels"]:
                if box["box_type"] in OCR_TYPES:
                    if "text" not in box:
                        box["text"] = None

        safe_save_json(dataset, output_file)
        print(f"Created new progress file: {output_file}")

    tasks = []
    total_blocks = 0

    for page in dataset:
        img_path = page["img_path"]

        if not os.path.exists(img_path):
            print(f"File not found: {img_path}. Skipping page.")
            continue

        for box in page["labels"]:
            if box["box_type"] in OCR_TYPES:
                total_blocks += 1
                if box.get("text") is None:
                    tasks.append((box, img_path))

    print(f"Total blocks of type {OCR_TYPES}: {total_blocks}")
    print(f"Remaining to process: {len(tasks)}")

    if not tasks:
        print("Everything already processed!")
        return dataset

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_info = {
            executor.submit(
                process_box_task, box, img_path, dataset, output_file
            ): (box, img_path)
            for (box, img_path) in tasks
        }

        for future in tqdm(
            as_completed(future_to_info),
            total=len(future_to_info),
            desc="OCR Progress"
        ):
            pass

    print(f"Done! Results saved to: {output_file}")
    return dataset

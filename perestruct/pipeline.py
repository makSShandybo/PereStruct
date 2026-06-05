import json
import tempfile
from pathlib import Path

from perestruct.yolo import get_yolo_boxes, visualize_boxes as viz_boxes
from perestruct.ocr import run_ocr
from perestruct.correction import run_correction
from perestruct.get_articles import ArticleAssembler


def parse_papers(
    images_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    yolo_weights: str = 'soviet_yolo.pt',
    keep_intermediate: bool = False
):
    """
    Complete processing pipeline: YOLO -> OCR -> GPT Correction -> Assembly.

    Args:
        images_dir: Path to folder with source images (.jpg, .jpeg).
        output_dir: Path to folder for saving final JSON and visualizations.
        threshold: Clustering threshold for article assembly (default 0.5).
        yolo_weights: YOLO weights filename.
        keep_intermediate: If True, saves intermediate JSON files
            (YOLO, OCR, Correction) to output_dir for debugging.
    """
    print("=" * 70)
    print("LAUNCHING FULL PROCESSING PIPELINE")
    print("=" * 70)

    images_path = Path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if keep_intermediate:
        path_yolo = output_path / "01_yolo_boxes.json"
        path_ocr = output_path / "02_ocr_result.json"
        path_corr = output_path / "03_corrected.json"
        path_final = output_path / "04_final_result.json"
        viz_dir = output_path / "visualizations"
    else:
        temp_dir = tempfile.gettempdir()
        path_yolo = Path(temp_dir) / "tmp_yolo_boxes.json"
        path_ocr = Path(temp_dir) / "tmp_ocr_result.json"
        path_corr = Path(temp_dir) / "tmp_corrected.json"
        path_final = output_path / "final_result.json"
        viz_dir = output_path / "visualizations"

    try:
        print("\n[1/4] Running YOLO detection...")
        yolo_boxes = get_yolo_boxes(str(images_path), model_weights=yolo_weights)

        if not yolo_boxes:
            raise ValueError("YOLO found no images or blocks.")

        with open(path_yolo, 'w', encoding='utf-8') as f:
            json.dump(yolo_boxes, f, ensure_ascii=False, indent=2)
        print(f"   Found {len(yolo_boxes)} pages")
        if keep_intermediate:
            print(f"   Saved: {path_yolo}")

        print("\n[2/4] Running OCR...")
        final_dataset = run_ocr(yolo_boxes, output_file=str(path_ocr))
        print("   OCR completed")
        if keep_intermediate:
            print(f"   Saved: {path_ocr}")

        print("\n[3/4] Running text correction...")
        run_correction(input_path=str(path_ocr), output_path=str(path_corr))
        print("   Correction completed")
        if keep_intermediate:
            print(f"   Saved: {path_corr}")

        print("\n[4/4] Assembling articles and generating visualizations...")
        assembler = ArticleAssembler(
            threshold=threshold,
            yolo_weights_name=yolo_weights
        )

        res = assembler.process_file(
            input_json_path=str(path_corr),
            output_json_path=str(path_final),
            viz_output_dir=str(viz_dir)
        )

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Final JSON: {path_final.resolve()}")
        print(f"Visualizations: {viz_dir.resolve()}")

        return res

    except Exception as e:
        print(f"\nCRITICAL PIPELINE ERROR: {e}")
        if not keep_intermediate:
            for p in [path_yolo, path_ocr, path_corr]:
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        raise e

    finally:
        if not keep_intermediate:
            for p in [path_yolo, path_ocr, path_corr]:
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass

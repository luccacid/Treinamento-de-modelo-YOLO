# -*- coding: utf-8 -*-
"""Executa o pipeline YOLO completo em um comando: organizar → treinar → validar.

Encadeia os scripts numerados, parando no primeiro que falhar (exit code != 0).
A limpeza (04_clear_dataset.py) fica de fora por ser destrutiva — rode à parte.

Uso:
    python run_pipeline.py
    python run_pipeline.py --epochs 50 --batch 8 --skip-organize
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent


def run(script: str, *args: str):
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    print(f"\n{'=' * 60}\n▶  {' '.join(cmd)}\n{'=' * 60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\n✗ Falhou em {script} (código {result.returncode}).")


def main():
    parser = argparse.ArgumentParser(description="Pipeline YOLO ponta a ponta")
    parser.add_argument("--data", default="data/dataset.yaml")
    parser.add_argument("--path", default="datasets")
    parser.add_argument("--model", default="yolov9m.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--name", default="license_plate_train")
    parser.add_argument("--skip-organize", action="store_true")
    args = parser.parse_args()

    if not args.skip_organize:
        run("01_organize_dataset.py", "--path", args.path)
    run("02_train_model.py", "--data", args.data, "--model", args.model,
        "--epochs", str(args.epochs), "--batch", str(args.batch),
        "--imgsz", str(args.imgsz), "--name", args.name)
    run("03_validate_model.py", "--data", args.data, "--imgsz", str(args.imgsz),
        "--model", f"runs/detect/{args.name}/weights/best.pt")
    print("\n✅ Pipeline concluído.")


if __name__ == "__main__":
    main()

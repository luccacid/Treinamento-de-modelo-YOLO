import shutil
from pathlib import Path
import argparse

def organize_files(base_dir: Path):
    """
    Move arquivos .jpg, .jpeg, .png para a pasta 'images'
    e arquivos .txt para a pasta 'labels'.
    """
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"

    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    img_extensions = {'.jpg', '.jpeg', '.png'}
    label_extension = '.txt'

    print(f"Processando pasta: {base_dir}")
    moved_imgs = 0
    moved_lbls = 0

    for file_path in base_dir.iterdir():
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()

        try:
            if ext in img_extensions:
                shutil.move(str(file_path), str(images_dir / file_path.name))
                moved_imgs += 1
            elif ext == label_extension:
                shutil.move(str(file_path), str(labels_dir / file_path.name))
                moved_lbls += 1
        except shutil.Error as e:
            print(f"Erro ao mover {file_path.name}: {e}")

    print(f"Movidos: {moved_imgs} imagens, {moved_lbls} labels.\n")

def main():
    parser = argparse.ArgumentParser(description="Organiza o dataset YOLO em pastas 'images' e 'labels'.")
    parser.add_argument(
        "--path",
        type=str,
        default="datasets",
        help="Caminho para a pasta principal do dataset (ex: 'datasets')."
    )
    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.is_dir():
        print(f"Erro: O diretório '{base_path}' não foi encontrado.")
        return

    data_splits = ['treino', 'validacao', 'teste']

    for split in data_splits:
        split_dir = base_path / split
        if split_dir.is_dir():
            organize_files(split_dir)
        else:
            print(f"Aviso: Pasta '{split_dir}' não encontrada, pulando.")

    print("✅ Estrutura YOLO organizada!")

if __name__ == "__main__":
    main()
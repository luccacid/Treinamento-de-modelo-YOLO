import argparse
import os
from pathlib import Path

def clear_files(directory: Path, extensions: list, dry_run: bool = False):
    """
    Deleta arquivos com extensões específicas dentro de um diretório.
    """
    if not directory.is_dir():
        print(f"Aviso: Diretório '{directory}' não encontrado, pulando.")
        return 0
    
    file_count = 0
    for file_path in directory.iterdir():
        # Deleta apenas se for um arquivo e tiver a extensão correta
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            print(f"  - Deletando {file_path.name}")
            if not dry_run:
                file_path.unlink()
            file_count += 1
    return file_count

def clear_caches(base_path: Path, dry_run: bool = False):
    """
    Encontra e deleta recursivamente todos os arquivos .cache.
    """
    print(f"\n--- 🔎 Procurando por caches em '{base_path}' ---")
    cache_files = list(base_path.rglob("*.cache"))
    
    if not cache_files:
        print("Nenhum arquivo .cache encontrado.")
        return 0
        
    for cache_file in cache_files:
        print(f"  - Deletando cache: {cache_file}")
        if not dry_run:
            cache_file.unlink()
    
    return len(cache_files)

def main():
    parser = argparse.ArgumentParser(
        description="Limpa imagens, labels e caches de um dataset YOLO."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="datasets",
        help="Caminho para a pasta principal do dataset (padrão: 'datasets')."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simula a exclusão sem deletar arquivos."
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Pula a confirmação de exclusão."
    )
    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.is_dir():
        print(f"Erro: O diretório '{base_path}' não foi encontrado.")
        return

    if args.dry_run:
        print("*** MODO DRY RUN ATIVADO (NENHUM ARQUIVO SERÁ DELETADO) ***")

    # 1. Confirmação de Segurança
    if not args.dry_run and not args.yes:
        print(f"\n⚠️ ATENÇÃO! ⚠️")
        print(f"Isso deletará permanentemente imagens, labels e caches dentro de:")
        print(f"{base_path.resolve()}")
        confirm = input("Você tem certeza que deseja continuar? (s/N): ")
        if confirm.lower() != 's':
            print("Operação cancelada.")
            return

    splits = ['treino', 'validacao', 'teste']
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    lbl_exts = ['.txt']
    
    total_deleted = 0

    # 2. Deletar Imagens e Labels
    for split in splits:
        print(f"\n--- 🧹 Limpando '{split}' ---")
        
        # Limpar imagens
        img_dir = base_path / split / 'images'
        print(f"Limpando {img_dir}...")
        count_img = clear_files(img_dir, img_exts, args.dry_run)
        print(f"{count_img} imagens deletadas.")
        total_deleted += count_img
        
        # Limpar labels
        lbl_dir = base_path / split / 'labels'
        print(f"Limpando {lbl_dir}...")
        count_lbl = clear_files(lbl_dir, lbl_exts, args.dry_run)
        print(f"{count_lbl} labels deletados.")
        total_deleted += count_lbl

    # 3. Deletar Caches
    count_cache = clear_caches(base_path, args.dry_run)
    total_deleted += count_cache
    
    # 4. Resumo Final
    print("\n" + "="*30)
    if args.dry_run:
        print(f"✅ DRY RUN Concluído. {total_deleted} arquivos teriam sido deletados.")
    else:
        print(f"✅ Limpeza Concluída. {total_deleted} arquivos foram deletados.")

if __name__ == "__main__":
    main()
import argparse

import torch
from ultralytics import YOLO


def validate(opt):
    """Carrega um modelo YOLO treinado e executa a validação."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    model = YOLO(opt.model)
    print(f"Iniciando validação com data='{opt.data}' e imgsz={opt.imgsz}...")
    results = model.val(
        data=opt.data,
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=device,
        name=opt.name,
    )
    print("Validação concluída.")
    print(results)

def main():
    parser = argparse.ArgumentParser(description="Script de Validação YOLO")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,  # Necessário para saber qual modelo validar
        help="Caminho para o modelo treinado (ex: runs/detect/train/weights/best.pt)"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/dataset.yaml",  # Padrão do projeto
        help="Caminho para o arquivo data.yaml"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=16, 
        help="Batch size (validação geralmente aceita mais que o treino)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640,  # Padronizado
        help="Tamanho da imagem para validação"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="validation_results", 
        help="Nome da pasta para salvar os resultados da validação"
    )
    
    args = parser.parse_args()
    validate(args)

if __name__ == "__main__":
    main()
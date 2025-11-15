import torch
import argparse
from ultralytics import YOLO

def validate(opt):
    """
    Carrega um modelo YOLO treinado e executa a validação.
    """
    try:
        # Detecta dispositivo (CUDA ou CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {device}")

        # Carrega o modelo
        model = YOLO(opt.model)

        # Inicia a validação
        print(f"Iniciando validação com data='{opt.data}' e imgsz={opt.imgsz}...")
        results = model.val(
            data=opt.data,
            imgsz=opt.imgsz,
            batch=opt.batch,
            device=device,
            name=opt.name  # Nome da pasta de resultados em 'runs/val/'
        )
        
        print("Validação concluída.")
        print(results)

    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

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
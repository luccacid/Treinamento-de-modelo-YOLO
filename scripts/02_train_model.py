import torch 
import argparse
from ultralytics import YOLO

def train_model(opt):

    try:
        # Detecta dispositivo (CUDA ou CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {device}")


        model = YOLO(opt.model)

        # Inicia o treinamento
        print(f"Iniciando treinamento com data='{opt.data}' para {opt.epochs} épocas...")
        model.train(
            data=opt.data,
            epochs=opt.epochs,
            imgsz=opt.imgsz,
            batch=opt.batch,
            device=device,
            name=opt.name  # Nome da pasta de resultados em 'runs/detect/'
            degrees=0.0,  # Sem rotação aleatória
            flipud=0.0,   # Sem flip vertical (upside-down)
            fliplr=0.0    # Sem flip horizontal (opcional, mas recomendado)
        )
        
        print("Treinamento concluído.")
        print(f"Resultados salvos em: runs/detect/{opt.name}")

    except Exception as e:
        print(f"Ocorreu um erro inesperado durante o treinamento: {e}")

def main():
    parser = argparse.ArgumentParser(description="Script de Treinamento YOLOv8")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov9m.pt", 
        help="Modelo base para começar (ex: yolov8n.pt) ou 'last.pt' para resumir."
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/dataset.yaml", 
        help="Caminho para o arquivo data.yaml"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=30, 
        help="Número de épocas para treinar"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=8, 
        help="Batch size (ajuste conforme a VRAM da sua GPU)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640, 
        help="Tamanho da imagem para treinamento"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="license_plate_train", 
        help="Nome da pasta para salvar os resultados do treino"
    )
    
    args = parser.parse_args()
    train_model(args)

if __name__ == "__main__":
    main()
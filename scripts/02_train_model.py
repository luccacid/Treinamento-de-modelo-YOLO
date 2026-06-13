import argparse

import torch
from ultralytics import YOLO


def train_model(opt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    model = YOLO(opt.model)
    print(f"Iniciando treinamento com data='{opt.data}' por {opt.epochs} épocas...")
    # Para placas de carro o objeto tem orientação fixa: desligamos rotação e
    # flips para não gerar amostras impossíveis (placa de cabeça para baixo).
    model.train(
        data=opt.data,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=device,
        name=opt.name,
        degrees=0.0,
        flipud=0.0,
        fliplr=0.0,
    )
    print(f"Treinamento concluído. Resultados em: runs/detect/{opt.name}")

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
        default=16, 
        help="Batch size (ajuste conforme a VRAM da sua GPU)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=1280, 
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
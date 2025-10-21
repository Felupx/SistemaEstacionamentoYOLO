import torch
import time
import cv2
from ultralytics import YOLO

# Carregar modelo YOLOv8 (pode ser 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.)
model = YOLO('yolov8s.pt')

# Imagem
img = 'images/teste.jpg'

tempo1 = time.time()
# Inferência
results = model(img)
tempo = time.time() - tempo1
print("Tempo decorrido: ", tempo)

# Salvar resultado com bounding boxes
for r in results:
    annotated_frame = r.plot()  # desenha caixas e labels na imagem
    cv2.imwrite('Resultado.jpg', annotated_frame)

# Analisar resultados
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Coordenadas da bbox
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Confiança
        confianca = float(box.conf[0])

        # Classe
        classe_id = int(box.cls[0])
        nomeClasse = model.names[classe_id]

        print(f"Classe: {nomeClasse} | Confiança: {confianca:.2f}")
        if nomeClasse == "car":
            print("############# Tem carro! #############")
        else:
            print("############# Não tem carro! #############")

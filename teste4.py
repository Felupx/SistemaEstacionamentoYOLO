import cv2
import time
from ultralytics import YOLO

# Carregar modelo YOLOv8
model = YOLO("yolov8s.pt")

# Abrir a câmera (0 = webcam padrão, se tiver mais câmeras pode testar 1, 2, ...)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    exit()

# Definir coordenadas da área (vaga)
# (x1, y1) canto superior esquerdo | (x2, y2) canto inferior direito
vaga_x1, vaga_y1 = 200, 200
vaga_x2, vaga_y2 = 400, 400

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tempo1 = time.time()
        results = model(frame, verbose=False)  # passa o frame da câmera para o modelo
        tempo = time.time() - tempo1

        vaga_ocupada = False

        # Pegar resultados
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

                # --- MUDANÇA PRINCIPAL AQUI ---
                # Verificar se tem carro ANTES de desenhar
                if nomeClasse == "car":

                    # 1. Mover o desenho da caixa para DENTRO do if
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{nomeClasse} {confianca:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 2. Manter a lógica de verificação da vaga
                    inter_x1 = max(vaga_x1, x1)
                    inter_y1 = max(vaga_y1, y1)
                    inter_x2 = min(vaga_x2, x2)
                    inter_y2 = min(vaga_y2, y2)

                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        vaga_ocupada = True

                # --- FIM DA MUDANÇA ---
                # As linhas de desenho que estavam aqui foram movidas para cima

        # Desenhar a área da vaga (isso continua igual)
        cor_vaga = (0, 0, 255) if vaga_ocupada else (0, 255, 0)
        cv2.rectangle(frame, (vaga_x1, vaga_y1), (vaga_x2, vaga_y2), cor_vaga, 3)
        texto = "OCUPADA" if vaga_ocupada else "LIVRE"
        cv2.putText(frame, f"Vaga: {texto}", (vaga_x1, vaga_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_vaga, 2)

        # Mostrar o vídeo com detecções
        cv2.imshow("Detecção YOLOv8", frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nPrograma interrompido pelo usuário.")

finally:
    cap.release()
    cv2.destroyAllWindows()
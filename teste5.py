import cv2
import time
from ultralytics import YOLO
from flask import Flask, jsonify
import threading

# ------------------------------
# Flask API
# ------------------------------
app = Flask(__name__)
vaga_ocupada = False  # variável compartilhada

@app.route('/status')
def status():
    global vaga_ocupada
    return jsonify({"vaga_ocupada": vaga_ocupada})

# ------------------------------
# Função de detecção
# ------------------------------
def detectar_vaga():
    global vaga_ocupada

    model = YOLO("yolov8s.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        return

    vaga_x1, vaga_y1 = 200, 200
    vaga_x2, vaga_y2 = 400, 400

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            vaga_ocupada = False

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confianca = float(box.conf[0])
                    classe_id = int(box.cls[0])
                    nomeClasse = model.names[classe_id]

                    if nomeClasse == "car":
                        inter_x1 = max(vaga_x1, x1)
                        inter_y1 = max(vaga_y1, y1)
                        inter_x2 = min(vaga_x2, x2)
                        inter_y2 = min(vaga_y2, y2)

                        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                            vaga_ocupada = True

            cor_vaga = (0, 0, 255) if vaga_ocupada else (0, 255, 0)
            texto = "OCUPADA" if vaga_ocupada else "LIVRE"
            cv2.rectangle(frame, (vaga_x1, vaga_y1), (vaga_x2, vaga_y2), cor_vaga, 3)
            cv2.putText(frame, f"Vaga: {texto}", (vaga_x1, vaga_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_vaga, 2)

            cv2.imshow("Detecção YOLOv8", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# ------------------------------
# Execução paralela
# ------------------------------
if __name__ == '__main__':
    # Thread para o sistema de detecção
    t = threading.Thread(target=detectar_vaga)
    t.daemon = True
    t.start()

    # Rodar servidor Flask
    app.run(host='0.0.0.0', port=5000)

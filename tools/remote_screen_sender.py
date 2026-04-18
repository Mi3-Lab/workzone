#!/usr/bin/env python3
import sys
import time
import datetime

try:
    import cv2
    import numpy as np
    from mss import mss
    from flask import Flask, Response
except ImportError:
    print("Dependencies missing! Run: pip install mss flask opencv-python")
    sys.exit(1)

app = Flask(__name__)
sct = mss()

def generate_frames():
    # Tenta usar o monitor principal (índice 1)
    # Se continuar preto, tente mudar para 0 ou 2
    monitor_idx = 1
    if len(sct.monitors) > monitor_idx:
        monitor = sct.monitors[monitor_idx]
    else:
        monitor = sct.monitors[0]
        
    print(f"[Sender] Iniciando captura do monitor {monitor_idx}: {monitor}")
    
    while True:
        try:
            # 1. Captura a tela
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            
            # 2. Converte BGRA para BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # 3. REDUZ RESOLUÇÃO (Essencial para rede rápida)
            # Vamos forçar 1280x720 para garantir que funcione
            frame = cv2.resize(frame, (1280, 720))
            
            # 4. ADICIONA "CORAÇÃO" (Texto para provar que o vídeo está vivo)
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-4]
            cv2.putText(frame, f"LIVE: {timestamp}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "WorkZone Remote Screen", (50, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # 5. Encodar com qualidade menor para latência zero
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ret: continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
        except Exception as e:
            print(f"Erro na captura: {e}")
            time.sleep(1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    import socket
    # Pega o IP real da rede
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        local_ip = s.getsockname()[0]
    except:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    
    print("="*50)
    print(f"SERVIDOR DE TELA WORKZONE ATIVO!")
    print(f"No Jetson, use: http://{local_ip}:5000/video_feed")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)

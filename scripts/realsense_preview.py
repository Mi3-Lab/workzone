import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import threading

class RealSenseCamera:
    """SOTA Threaded Camera Reader para RealSense"""
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.latest_frame = None
        self.running = False
        self.lock = threading.Lock()
        
    def start(self):
        try:
            # Seleciona o stream RGB
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            profile = self.pipeline.start(self.config)
            
            # [SOTA FIX]: Desativar "Auto-Exposure Priority"
            # Isso impede que a câmera baixe o FPS em ambientes escuros para capturar luz.
            color_sensor = profile.get_device().query_sensors()[1] # O sensor RGB geralmente é o index 1
            if color_sensor.supports(rs.option.auto_exposure_priority):
                color_sensor.set_option(rs.option.auto_exposure_priority, 0)
                
            print(f"[HW] RealSense inicializada a {self.width}x{self.height} @ {self.fps} FPS")
            
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Erro ao iniciar a RealSense: {e}")
            return False

    def _update(self):
        """Loop na thread de fundo para extrair frames na velocidade máxima"""
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                # Converte e guarda o frame mais recente em memória protegida
                frame_data = np.asanyarray(color_frame.get_data())
                with self.lock:
                    self.latest_frame = frame_data
            except Exception as e:
                print(f"[Erro na Thread RealSense] {e}")

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.pipeline.stop()

def main():
    # Inicializa câmera em alta resolução
    # Dica: RealSense suporta 1280x720 @ 30fps nativamente muito bem
    cam = RealSenseCamera(width=1280, height=720, fps=30)
    if not cam.start():
        return

    output_dir = os.path.join(os.getcwd(), "captured_images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n--- SOTA RealSense Viewer ---")
    print(" - [ESPAÇO]: Iniciar contagem regressiva de 5s para Foto")
    print(" - [ F ]: Alternar Tela Cheia (Fullscreen)")
    print(" - [ Q ] ou [ESC]: Sair")

    window_name = 'SOTA RealSense Preview'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    is_fullscreen = False

    countdown_start_time = None
    counting_down = False
    
    # Variáveis para FPS
    prev_time = time.time()
    fps_val = 0
    fps_smooth = 30.0

    try:
        while True:
            start_loop = time.time()
            
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.01) # Previne uso de 100% da CPU
                continue

            display_image = frame
            
            # FPS Calculation (EMA Smooth)
            curr_time = time.time()
            fps_val = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            fps_smooth = (fps_smooth * 0.9) + (fps_val * 0.1)

            # Draw HUD
            cv2.putText(display_image, f"FPS: {fps_smooth:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if counting_down:
                elapsed = time.time() - countdown_start_time
                remaining = 5 - int(elapsed)
                
                if remaining > 0:
                    text = f"Foto em {remaining}s"
                    font = cv2.FONT_HERSHEY_DUPLEX
                    scale = 2
                    color = (0, 0, 255) # Red
                    thickness = 4
                    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
                    x = (display_image.shape[1] - text_size[0]) // 2
                    y = (display_image.shape[0] + text_size[1]) // 2
                    
                    cv2.putText(display_image, text, (x+2, y+2), font, scale, (0, 0, 0), thickness)
                    cv2.putText(display_image, text, (x, y), font, scale, color, thickness)
                else:
                    # Salva a imagem original (1280x720) sem o HUD do countdown
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(output_dir, f"rs_hq_{timestamp}.jpg")
                    cv2.imwrite(filename, frame) # Usamos 'frame' ao invés de 'display_image' para não salvar o texto
                    print(f"\n[SUCESSO] Imagem de alta qualidade salva: {filename}")
                    
                    # Efeito de Flash na tela
                    display_image = np.full_like(display_image, 255)
                    cv2.imshow(window_name, display_image)
                    cv2.waitKey(100)
                    
                    counting_down = False

            # Exibição SOTA (Suave e sem bloqueio)
            cv2.imshow(window_name, display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            elif key == ord(' ') and not counting_down:
                print("Iniciando contagem de 5 segundos...")
                countdown_start_time = time.time()
                counting_down = True

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Finalizado.")

if __name__ == "__main__":
    main()

import cv2
import imageio
from ultralytics import YOLO
from utils import process_frame, draw_detections

def main():
    # Загрузка модели
    model = YOLO('models/yolo11s.pt')
    
    # Путь к входному видео
    video_path = "data/2025-02-27 18-59-26.mp4"
    cap = cv2.VideoCapture(video_path)
    
    frames = []  # Список для сохранения кадров для GIF
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Выполнение инференса с использованием модели YOLO от ultralytics
        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy()  # Получаем координаты, confidence и класс

        # Отрисовка результатов на кадре
        annotated_frame = draw_detections(frame.copy(), detections)
        
        # Добавляем кадр для GIF (конвертируем цвет из BGR в RGB)
        frames.append(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        
        # Отображаем результат в окне
        cv2.imshow("Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Сохранение GIF
    output_path = "data/output.gif"
    imageio.mimsave(output_path, frames, fps=10)
    print(f"GIF сохранён: {output_path}")

if __name__ == "__main__":
    main()

import cv2

def process_frame(frame, model):
    """
    Обрабатывает кадр с помощью модели и возвращает результаты детекции.
    Предполагается, что model возвращает объект, у которого результаты можно получить через results.xyxy[0]
    """
    results = model(frame)
    # Приводим результаты к формату numpy массива
    detections = results.xyxy[0].cpu().numpy()
    return detections

def draw_detections(frame, detections):
    """
    Отрисовывает bounding boxes и информацию о детекции на кадре.
    detections: numpy массив, где каждая строка - [x1, y1, x2, y2, confidence, class]
    """
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        # Приводим координаты к целым числам
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        
        if conf > 0.6:
        # Рисуем прямоугольник
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Подготавливаем текст с информацией: класс и вероятность
            label = f"Человек: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 
                        0.5, (0, 255, 0), 2)
    return frame

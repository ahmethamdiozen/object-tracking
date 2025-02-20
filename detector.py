import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision import Detections
from supervision import BoxAnnotator


class ObjectDetection:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device:", self.device)
        
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = BoxAnnotator(color=ColorPalette.DEFAULT, thickness=3)
    
    def load_model(self):
        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
        try:
            model.fuse()
        except Exception as e:
            print(f"Model fuse işlemi başarısız: {e}")
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results
    
    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []
        
        # Extract detections for person class
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.cpu().numpy())
                if class_id == 0:
                    xyxys.append(box.xyxy.cpu().numpy())
                    confidences.append(box.conf.cpu().numpy())
                    class_ids.append(class_id)
        
        # Setup detections for visualization
        detections = Detections(
            xyxy=np.array(xyxys).reshape(-1, 4),
            confidence=np.array(confidences).reshape(-1),
            class_id=np.array(class_ids).reshape(-1)
        )
        
        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                       for class_id, confidence in zip(detections.class_id, detections.confidence)]
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        
        return frame
    
    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        if not cap.isOpened():
            print("Hata: Kamera açılamadı!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            start_time = time()
            
            ret, frame = cap.read()
            if not ret:
                print("Hata: Kamera karesi alınamadı!")
                break
            
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            
            end_time = time()
            fps = 1 / (end_time - start_time)
            
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        

detector = ObjectDetection(capture_index=0)
detector()

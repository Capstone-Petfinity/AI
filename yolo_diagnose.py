from ultralytics import YOLO
from PIL import  Image,ImageDraw, ImageFont
import cv2

def add_text_to_image(image, text, position=(10, 10), font_size=10):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./godic.ttf", font_size)  
    draw.text(position, text, fill="red", font=font)
    return image

def yolo_detection_inference(img, model):
    
    custom_labels={0: '정상', 1: '슬개골 탈구'}
    
    model=YOLO(model)
    
    results = model(img)
    
    res_plotted = results[0].plot()
    
    disease_name='정상'
    
    for i, box in enumerate(results[0].boxes):
        class_id = int(box.cls)  # 클래스 ID 가져오기
        if class_id in custom_labels:
            results[0].names[class_id] = custom_labels[class_id]
            
            if class_id ==1:
                disease_name = '슬개골 탈구'
                
    res_plotted = results[0].plot()
    res_plotted = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    confidence=None
    
    return res_plotted, disease_name, confidence

def yolo_segmentation_inference(img, model, disease):
    
    custom_labels={0: '정상', 1: disease}
    disease_name = '정상'
    
    model=YOLO(model)
    
    results = model(img)
    
    for i, box in enumerate(results[0].boxes):
        class_id = int(box.cls)  # 클래스 ID 가져오기
        if class_id in custom_labels:
            
            results[0].names[class_id] = custom_labels[class_id]
            
            if class_id ==1:
                disease_name = custom_labels[1]
    
    res_plotted = results[0].plot()
    res_plotted = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    confidence=None
    
    return res_plotted, disease_name, confidence

def yolo_classification_inference(img, model):
    
    custom_labels={0: '정상', 1: '색소침착성각막염', 2: '결막염/비궤양각막질환', 3: '궤양성각막질환', 4: '백내장'}
    
    model=YOLO(model)
    
    results = model(img)
    
    confidence = results[0].probs.top1conf.item()  
    class_id = results[0].probs.top1  
    disease_name = custom_labels[class_id]
    
    image=Image.open(img)
    res_plotted= add_text_to_image(image, f"{disease_name} ({confidence:.2f})")
    
    return res_plotted, disease_name, confidence
    
    




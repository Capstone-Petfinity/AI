from ultralytics import YOLO
from PIL import  ImageDraw, ImageFont

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
    confidence=None
    
    return res_plotted, disease_name, confidence

def yolo_segmentation_inference(img, model):
    
    custom_labels={0: '정상', 1: '기관 허탈'}
    disease_name = '정상'
    
    model=YOLO(model)
    results = model(img)
    
    for i, box in enumerate(results[0].boxes):
        class_id = int(box.cls)  # 클래스 ID 가져오기
        if class_id in custom_labels:
            
            results[0].names[class_id] = custom_labels[class_id]
            
            if class_id ==1:
                disease_name = '기관 허탈'
    
    res_plotted = results[0].plot()
    confidence=None
    
    return res_plotted, disease_name, confidence

def yolo_classification_inference(img, model):
    
    custom_labels={0: '정상', 1: '색소침착성각막염', 2: '결막염/비궤양각막질환', 3: '궤양성각막질환', 4: '백내장'}
    
    model=YOLO(model)
    results = model(img)
    
    confidence = results[0].probs.data.max.item()  # 가장 높은 컨피던스 값
    class_id = results[0].probs.data.argmax().item()  # 가장 높은 컨피던스를 가진 클래스 ID
    disease_name = custom_labels[class_id]
    
    res_plotted= add_text_to_image(img, f"{disease_name} ({confidence:.2f})")
    
    return res_plotted, disease_name, confidence
    
    




from flask import Flask, jsonify, request, send_file
import os
import requests
from io import BytesIO
from PIL import Image
from .yolo_diagnose import yolo_detection_inference
app = Flask(__name__)

@app.route('/ai_diagnose', methods=['POST'])
def diagnose():
    image_url = request.form.get('image_url')
    response = requests.get(image_url)
    
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image=image.resize((640,640))
        image.save("./requested_image.jpg",format='JPEG')
    
    else:
        return("No URL")
    
    user_uuid= request.form.get('user_uuid')
    disease_area= request.form.get('disease_area')
    type= request.form.get('type')
    position= request.form.get('position')
    disease= request.form.get('disease')
    
    model= "./yolo_models/skeletal/mu05(detection).pt"
    img_path="./requested_image.jpg"
    
    res_plotted, disease_name, confidence=yolo_detection_inference(image, model)
    res_plotted.save("./result_image.jpg",format='JPEG')
    
    files = {
        'file': ('BBong.jpg', open("./result_image.jpg", 'rb'), 'image/jpg'),
    }
    
    url = 'https://blog-back.donghyuns.com/upload/image/post'

    # HTTP POST 요청 보내기
    response = requests.post(url, files=files)
    response_data = response.json()
    insert_id = response_data.get("insertId")
    
    
    diagnose_result={
        "user_uuid" :str(user_uuid),
        "disease_name":disease_name,
        "percent" :str(confidence),
        "insert_id ":str(insert_id) ,
    }
    
    return jsonify(diagnose_result)

if __name__ == '__main__':
    # Flask 애플리케이션을 0.0.0.0 호스트와 5000 포트로 실행
    app.run(host='0.0.0.0', port=5000)
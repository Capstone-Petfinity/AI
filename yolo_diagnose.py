from flask import Flask, render_template, request, jsonify
from PIL import Image
import os
from ultralytics import YOLO
import torch
import io
import base64

app = Flask(__name__)

@app.route('/hello')
def hello():
    return 'Hello, World!'

@app.route('/diagnose_ai', methods=['POST', 'GET'])
def diagnose_ai():
    if (request.method == 'POST'):
        
        #user_type = request.form.get('user_type')
        disease_area = request.form.get('disease_area')
        type = request.files.get('type')
        disease = request.form.get('disease')
        img = request.files.get('img')
                    
        img=Image.open(img.stream)
        img=img.resize(640,640)
        
        model_name="{:s}_{:s}_{:s}.pt".format(disease_area,type,disease)
        
        if(os.path.isfile("./yolo_models/"+model_name)):
            model=YOLO("./yolo_models/"+model_name)
            results = model(img)
            res_plotted = results[0].plot()
            
            cls_name=None
            
            if results[0].masks is not None:
                for counter, _ in enumerate(results[0].masks.data):
                    cls_id = int(results[0].boxes[counter].cls.item())
                    cls_name = model.names[cls_id]
            
        result_img = io.BytesIO()
        res_plotted.save(result_img, 'PNG')
        result_img.seek(0)

        img_base64 = base64.b64encode(result_img.getvalue()).decode('utf-8')
        
        response = {
            'image': img_base64,
            'info': cls_name
        }
        
        return jsonify(response)
    
    else :
        return "No FormData"
        
if __name__ == '__main__':
    # Flask 애플리케이션을 0.0.0.0 호스트와 20002 포트로 실행
    app.run(host='0.0.0.0', port=20002, debug=True)


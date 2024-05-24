from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import os
import io
import base64

app = Flask(__name__)

CORS(app)  # 모든 도메인에 대해 CORS 허용

@app.route('/hello')
def hello():
    return jsonify(message="Hello, World!")

@app.route('/formdata_test')
def formdata_test():
    if (request.method == 'POST'):
        
        user_type = request.form.get('user_type')
        disease_area = request.form.get('disease_area')
        type = request.form.get('type')
        disease = request.form.get('disease')
        img = request.files.get('img')
                    
        img=Image.open("./BBong.jpg")
        img=img.resize(640,640)
        
            
        result_img = io.BytesIO()
        img.save(result_img, 'PNG')
        result_img.seek(0)

        img_base64 = base64.b64encode(result_img.getvalue()).decode('utf-8')
        
        response = {
            'image': img_base64,
            'info': "Test Success!",
            'user_type': user_type,
            "disease_area":disease_area,
            "type":type,
            "disease":disease,
        }
        
        return jsonify(response)
    
    else :
        return "No FormData"

if __name__ == '__main__':
    # Flask 애플리케이션을 0.0.0.0 호스트와 5000 포트로 실행
    app.run(host='0.0.0.0', port=5000)
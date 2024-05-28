from flask import Flask, jsonify, request, send_file
import os
import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/ai_diagnose', methods=['POST'])
def diagnose():
    image_url = request.form.get('image_url')
    response = requests.get(image_url)
    
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image=image.resize((640,640))
    
    else:
        return("No URL")
    
    user_uuid= request.form.get('user_uuid')
    disease_area= request.form.get('disease_area')
    type= request.form.get('type')
    position= request.form.get('position')
    disease= request.form.get('disease')
    


if __name__ == '__main__':
    # Flask 애플리케이션을 0.0.0.0 호스트와 5000 포트로 실행
    app.run(host='0.0.0.0', port=5000)
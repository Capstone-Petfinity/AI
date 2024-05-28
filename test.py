from flask import Flask, jsonify, request, send_file
import os
import requests

app = Flask(__name__)

@app.route('/formdata_test')
def formdata_test():

    # 파일 경로
    file_path = "./BBong.jpg"

    # FormData에 포함될 파일 정보
    files = {
        'file': ('BBong.jpg', open(file_path, 'rb'), 'image/jpg')
    }

    # 요청을 보낼 URL
    url = 'https://blog-back.donghyuns.com/upload/image/post'

    # HTTP POST 요청 보내기
    response = requests.post(url, files=files)

    response_data = response.json()
    insert_id = response_data.get("insertId")
    
    print(f'Status Code: {response.status_code}')
    print(f'Status Text: {response.text}')
    print(f'Response Headers: {response.headers}')
    
    return jsonify(status_code=response.status_code, response_text=response.text, insert_id=insert_id)

@app.route('/get_image',methods=['POST'])
def get_image():

if __name__ == '__main__':
    # Flask 애플리케이션을 0.0.0.0 호스트와 5000 포트로 실행
    app.run(host='0.0.0.0', port=5000)
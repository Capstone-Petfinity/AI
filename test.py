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
    # 클라이언트로부터 insertId를 받아옴
    insert_id = request.form.get('insertId')

    if not insert_id:
        return jsonify(message="insertId is required"), 400

    # 요청을 보낼 URL
    url = 'https://blog-back.donghyuns.com/post/url'

    response = requests.post(url, json={"postSeq": insert_id})

    print(f'Status Code: {response.status_code}')
    print(f'Response Text: {response.text}')
    
    if response.status_code == 200:
        # 서버로부터 받은 URL 추출
        response_data = response.json()
        image_url = response_data.get("result", [None])[0]

        if image_url:
            return jsonify(status_code=response.status_code, image_url=image_url)
        else:
            return jsonify(message="Failed to retrieve image URL from response"), 502
    else:
        return jsonify(message="Failed to retrieve image URL", status_code=response.status_code, response_text=response.text)

if __name__ == '__main__':
    # Flask 애플리케이션을 0.0.0.0 호스트와 5000 포트로 실행
    app.run(host='0.0.0.0', port=5000)
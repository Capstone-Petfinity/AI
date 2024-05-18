from flask import Flask, render_template
from PIL import Image
import os
from flask import Flask

app = Flask(__name__)

@app.route('/hello')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    # Flask 애플리케이션을 0.0.0.0 호스트와 20002 포트로 실행
    app.run(host='0.0.0.0', port=20002)


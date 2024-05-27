from flask import Flask, jsonify, request
import os
import requests

app = Flask(__name__)

@app.route('/formdata_test')
def formdata_test():
    if (request.method == 'POST'):
        
        user_uuid=request.form.get('user_uuid')
        user_type = request.form.get('user_type')
        disease_area = request.form.get('disease_area')
        type = request.form.get('type')
        position=request.form.get('position')
        disease = request.form.get('disease')
        insert_id = request.form.get('insert_id')
                    
        
        
        return jsonify(response)
    
    else :
        return "No FormData"

if __name__ == '__main__':
    # Flask 애플리케이션을 0.0.0.0 호스트와 5000 포트로 실행
    app.run(host='0.0.0.0', port=5000)
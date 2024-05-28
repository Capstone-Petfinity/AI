from flask import Flask, jsonify, request
import requests
from io import BytesIO
from PIL import Image
from .yolo_diagnose import (
    yolo_detection_inference,
    yolo_segmentation_inference,
    yolo_classification_inference,
)
from .torch_classification_diagnose import efficientnet_inference
from .torch_detection_diagnose import fasterrcnn_inference

app = Flask(__name__)

disease_dic = {
    "eb07": "결막염",
    "eb08": "궤양성각막질환",
    "eb09": "비궤양성각막질환",
    "eb05": "색소침착성결막염",
    "eb10": "백내장",
    "eb11": "유리체변성",
    "sk01": "구진플라크",
    "sk02": "비듬각질상피성잔고리",
    "sk03": "태선화과다색소침착",
    "sk04": "농포여드름",
    "sk05": "미란궤양",
    "sk06": "결절종괴",
    "ab04": "복부 종양",
    "ab05": "결석",
    "ab09": "복수",
    "ch02": "기관허탈",
    "ch03": "종격동변위",
    "ch04": "흉강종양",
    "ch05": "기흉",
    "ch06": "횡경막 탈장",
    "mu05": "슬개골탈구",
}

eye_model_path = {
    "US": "./yolo_models/eye/US(classification).pt",
    "CM": "./yolo_models/eye/CM(classification).pt",
}

Lateral_model_path = {
    "ch04": "./torch_models/chest/Lateral/ch04(efficientv2).pt",
    "ch05": "./torch_models/chest/Lateral/ch05(efficientv2).pt",
    "ab04": "./torch_models/stomach/Lateral/ab04(efficientv2).pt",
    "ab05": "./torch_models/stomach/Lateral/ab05(FasterRCNN).pt",
    "ab09": "./torch_models/stomach/Lateral/ab09(efficientv2).pt",
}

VD_model_path = {
    "ch04": "./torch_models/chest/VD/ch04(efficientv2).pt",
    "ch05": "./torch_models/chest/VD/ch05(efficientv2).pt",
    "ab04": "./torch_models/stomach/VD/ab04(efficientv2).pt",
    "ab05": "./torch_models/stomach/VD/ab05(FasterRCNN).pt",
    "ab09": "./torch_models/stomach/VD/ab09(efficientv2).pt",
}

chest_model_path = {"ch02": "./yolo_models/chest/ch02(segmentation).pt"}


@app.route("/diagnosis", methods=["POST"])
def diagnose():
    data = request.get_json()
    image_url = data.get("img_url")
    response = requests.get(image_url)

    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image = image.resize((640, 640))
        image.save("./requested_image.jpg", format="JPEG")
    else:
        return "No URL", 400

    user_uuid = data.get("user_uuid")
    disease_area = data.get("disease_area")
    type = data.get("type")
    position = data.get("position")
    disease = data.get("disease")

    disease_name = disease_dic.get(disease)

    # 모델 선택 로직
    if disease == "ch02":
        model_path = chest_model_path.get(disease)
    elif disease_area == "eye":
        model_path = eye_model_path.get(type)
    elif disease_area == "chest":
        if position == "Lateral":
            model_path = Lateral_model_path.get(disease)
        elif position == "VD":
            model_path = VD_model_path.get(disease)
    else:
        # 기본 모델 설정
        model_path = "./yolo_models/skeletal/mu05(detection).pt"

    if not model_path:
        return "No model found for the given parameters", 400

    # task 결정 및 함수 호출
    img_path = "./requested_image.jpg"
    if "efficientv2" in model_path:
        res_plotted, detected_disease_name, confidence = efficientnet_inference(
            img_path, model_path, disease_name
        )
    elif "fasterRCNN" in model_path:
        res_plotted, detected_disease_name, confidence = fasterrcnn_inference(
            img_path, model_path
        )
    elif "detection" in model_path:
        res_plotted, detected_disease_name, confidence = yolo_detection_inference(
            img_path, model_path
        )
    elif "segmentation" in model_path:
        res_plotted, detected_disease_name, confidence = yolo_segmentation_inference(
            img_path, model_path
        )
    elif "classification" in model_path:
        res_plotted, detected_disease_name, confidence = yolo_classification_inference(
            img_path, model_path
        )

    res_plotted.save("./result_image.jpg", format="JPEG")

    files = {
        "file": ("result_image.jpg", open("./result_image.jpg", "rb"), "image/jpg"),
    }

    url = "https://blog-back.donghyuns.com/upload/image/post"

    # HTTP POST 요청 보내기
    response = requests.post(url, files=files)
    response_data = response.json()
    insert_id = response_data.get("insertId")

    if detected_disease_name == "결막염/비궤양각막질환":
        content = "결막염은 눈이 충혈되고 눈꼽이 나는 것이 특징입니다. 비궤양각막질환은 눈꼽은 없으나 충혈이 되는 경우가 있으며 시력이 저하될 수 있습니다."
    
    else:
        content = None
        
    diagnose_result = {
        "user_uuid": str(user_uuid),
        "disease_name": detected_disease_name,
        "percent": str(confidence),
        "insert_id": str(insert_id),
        "content" : content
    }

    return jsonify(diagnose_result)


if __name__ == "__main__":
    # Flask 애플리케이션을 0.0.0.0 호스트와 5000 포트로 실행
    app.run(host="0.0.0.0", port=5000)
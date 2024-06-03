from flask import Flask, jsonify, request
import requests
from io import BytesIO
from PIL import Image
from yolo_diagnose import (
    yolo_detection_inference,
    yolo_segmentation_inference,
    yolo_classification_inference,
)
from torch_classification_diagnose import efficientnet_inference
from torch_detection_diagnose import fasterrcnn_inference
from skin_diagnose import skin_classification_inference
import logging

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
    "ch06": "./torch_models/chest/Lateral/ch06(efficientv2).pt",
    "ab04": "./torch_models/stomach/Lateral/ab04(efficientv2).pt",
    "ab05": "./torch_models/stomach/Lateral/ab05(FasterRCNN).pt",
    "ab09": "./torch_models/stomach/Lateral/ab09(efficientv2).pt",
}

VD_model_path = {
    "ch04": "./torch_models/chest/VD/ch04(efficientv2).pt",
    "ch05": "./torch_models/chest/VD/ch05(efficientv2).pt",
    "ch06": "./torch_models/chest/VD/ch06(efficientv2).pt",
    "ab04": "./torch_models/stomach/VD/ab04(efficientv2).pt",
    "ab05": "./torch_models/stomach/VD/ab05(FasterRCNN).pt",
    "ab09": "./torch_models/stomach/VD/ab09(efficientv2).pt",
}

chest_model_path = {"ch02": "./yolo_models/chest/ch02(segmentation).pt",
                    "ch03": "./yolo_models/chest/ch03(segmentation).pt"}

skin_content_dic={"구진(플라크)":"구진이란 염증성 여드름 병변과 비염증성 여드름 병변의 중간 형태이며 피부의 단단한 덩어리로 직경은 0.5cm ~ 1cm 정도 입니다. 작고 딱딱한 붉은 색의 병변이나 안에 고름은 잡히지 않은 상태로 나타납니다.",
        "비듬/각질/상피성잔고리": "비듬/각질은 피부 표피에서 건조하고 얇은 형태로 나타나는 표피세포로 생명력이 없는 죽은 세포입니다. 농포나 수포가 파열된 후 나타나는 원형으로 나열된 비듬/각질을 상피성잔고리라고 부릅니다.",
        "태선화 과다색소침착":"태선화 과다색소침착은 피부가 두꺼워지고 단단해져 피부주름이 뚜렷해지는 증상을 동반하며 피부색이 검게 변하는 질환이며 색소침착의 일반적인 원인으로는 피부 멜라닌 색소 과다, 비정상적 피부 성장, 피부 염증 후 색소침착 등이 있습니다.",
        "농포(여드름)": "농포란 염증성 여드름 병변으로 고름이 차있는 융기된 주머니를 말합니다. 농포의 전 단계인 구진과 비슷한 크기로 작고 둥그런 모양이지만 안에 고름을 포함하고 있는 것이 구진과 다른 점입니다.  손으로 짜게 되면 염증을 유발할 수 있으므로 손대지 않는 것이 좋습니다.",
        "궤양/미란" :"피부나 점막 상피조직의 부분적 결손이 피하조직이나 점막하조직에 까지 이른 상태를 궤양이라고 합니다. 결손이 피부 표층에 국한될 경우 미란이라고 부릅니다.",
        "결절(종괴)" : "결절은 피부 안쪽이나 밑에 딱딱하고 솟아오른 조직이나 유체를 의미합니다. 경우에 따라 가려움증을 동반할 수 있으며  위치와 원인에 따라 유선 종양, 지방종, 피지 낭종, 비만세포증 질환이 원인이 될 수 있습니다. 병원에 내원하여 악성/양성 여부 검사가 필요합니다.",
        "정상" : "정상입니다.",}

eye_content_dic={"색소침착성각막염":"건조증으로 인해 장기간 부족한 눈물량이 지속되면서 만성적인 자극이 각막에 주어지고, 그로 인해 각막에 색소가 침착되는 증상을 의미합니다.",
        "궤양성각막질환":"궤양성각막질환은 눈 앞부분의 투명한 조직인 각막에 손상이 있을 때 세균이나 바이러스, 진균 또는 여러 가지 원인 등에 의해서 염증이 발생하고 이에 따라 각막의 일부가 움푹 파이는 질병입니다. 세균으로 인한 궤양성각막질환의 경우 극심한 안구 통증, 충혈, 눈부심 등을 동반할 수 있습니다.",
        "백내장":"백내장이란 카메라의 렌즈에 해당하는 눈 속의 수정체가 뿌옇게 혼탁해져서 시력장애가 발생하는 질환이며, 백내장이 발생한 위치와 정도에 따라 다양한 모습으로 보입니다. 백내장의 원인으로는 노화, 당뇨병, 영양 결핍, 유전적 소인 등이 있습니다.",
        "결막염/비궤양성각막질환":"결막염은 눈꺼풀의 안쪽과 흰 눈의 표면(공막)으로 이루어진 결막에 염증이 생긴 질병을 의미합니다. 비궤양성각막질환은 바이러스, 세균 등에 의해 각막에 염증이 발생하되 각막에 궤양(표면이 움푹파이는 현상)이 발생하지 않는 질환입니다. 결막염은 눈이 충혈되고 눈꼽이 나는 것이 특징입니다. 비궤양각막질환은 눈꼽은 없으나 충혈이 되는 경우가 있으며 시력이 저하될 수 있습니다.",
        "유리체변성":"유리체변성이란 유리체가 액화가 되는 것으로 유리체의 액체와 고체 성분이 서로 다른 부분으로 분리돼 발생합니다. 유리체변성은 대부분 노화에 의한 것이 가장 많으며 그 이외에 감염, 외상, 포도막염 등에 의해서도 발생할 수 있습니다.",
        "정상" : "정상입니다.",}

@app.route("/diagnosis", methods=["POST"])
def diagnose():
    
    logging.info("====================Start====================")
    
    data = request.get_json()
    image_url = data.get("img_url")
    response = requests.get(image_url)
    
    logging.info(f"Request data: {data}")
    
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image = image.resize((640, 640))
        image = image.convert('RGB')
        image.save("./requested_image.jpg", format="JPEG")
    else:
        logging.info("NO URL")
        return "No URL", 400

    user_uuid = data.get("user_uuid")
    disease_area = data.get("disease_area")
    type = data.get("type")
    position = data.get("position")
    disease = data.get("disease")

    logging.info(f"user_uuid: {user_uuid}, disease_area: {disease_area}, type: {type}, position: {position}, disease: {disease}")
    
    disease_name = disease_dic.get(disease)

    # 모델 선택 로직
    if disease == "ch02" or disease == "ch03":
        model_path = chest_model_path.get(disease)
    elif disease_area == "skin":
        model_path = "./torch_models/skin/skin.pth"
    elif disease_area == "eye":
        model_path = eye_model_path.get(type)
    elif disease_area == "chest" or disease_area == "stomach":
        if position == "Lateral"  :
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
    elif "skin" in model_path:
        res_plotted, detected_disease_name, confidence = skin_classification_inference(
            img_path, model_path
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
            img_path, model_path, disease_name
        )
    elif "classification" in model_path:
        res_plotted, detected_disease_name, confidence = yolo_classification_inference(
            img_path, model_path, (type=='US')
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
    content = None
    
    if disease_area == "eye":
        content =  eye_content_dic[detected_disease_name]
        
    
    elif disease_area == "skin":
        content = skin_content_dic[detected_disease_name]
    
    if confidence == None:
        confidence = None
    else:
        confidence=confidence*100
        confidence=round(confidence,2)

    diagnose_result = {
        "user_uuid": str(user_uuid),
        "disease_name": detected_disease_name,
        "percent": str(confidence),
        "insert_id": str(insert_id),
        "content" : content
    }

    logging.info(f"Result data: {diagnose_result}")
    
    return jsonify(diagnose_result)


if __name__ == "__main__":
    # Flask 애플리케이션을 0.0.0.0 호스트와 5000 포트로 실행
    app.run(host="0.0.0.0", port=5000)
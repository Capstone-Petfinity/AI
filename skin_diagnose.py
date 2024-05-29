import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageDraw, ImageFont

def add_text_to_image(image, text, position=(10, 10), font_size=10):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(
        "./godic.ttf", font_size
    )  # Ensure 'arial.ttf' is available
    draw.text(position, text, fill="red", font=font)
    return image

disease_name = {
        0: "구진(플라크)",
        1: "비듬/각질/상피성잔고리",
        2: "태선화 과다색소침착",
        3: "농포(여드름)",
        4: "궤양/미란",
        5: "결절(종괴)",
        6: "정상",
    } 


def skin_classification_inference(img_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    num_classes = len(disease_name)  # 클래스 수를 설정
    model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 이미지 전처리
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(img_path).convert("RGB")
    image=image.resize((640,640))
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 모델 추론
    with torch.no_grad():
        outputs = model(input_batch)

    # 결과 해석
    _, predicted = torch.max(outputs, 1)
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0, predicted].item()
    detected_disease_name = disease_name[predicted.item()]

    res_plotted = add_text_to_image(
        image, f"{detected_disease_name} ({confidence:.2f})", font_size = 20
    )

    return res_plotted, detected_disease_name, confidence



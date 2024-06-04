import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch

import torchvision
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CustomDataset(Dataset):
    def __init__(self, img_path,transforms=None):
        self.transforms = transforms
        self.imgs = [img_path]

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        height, width = img.shape[0], img.shape[1]

        
        if self.transforms is not None:
            transformed = self.transforms(image=img)
            img = transformed["image"]
        file_name = img_path.split('/')[-1]
        return file_name, img, width, height

    def __len__(self):
        return len(self.imgs)
    
def get_test_transforms():
    return A.Compose([
        A.Resize(640,640),
        ToTensorV2(),
    ])

def build_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def box_denormalize(x1, y1, x2, y2, width, height, confidence):
    x1 = (x1 / 640) * width
    y1 = (y1 / 640) * height
    x2 = (x2 / 640) * width
    y2 = (y2 / 640) * height
    return int(x1), int(y1), int(x2), int(y2), float(confidence)

def add_text_to_image(image, text, position, font_size):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./godic.ttf", font_size)  # Ensure 'godic.ttf' is available
    draw.text(position, text, fill="red", font=font)
    return image

def inference(model, test_loader, device, threshold=0.6):
    model.eval()
    model.to(device)
    
    results = []
    for _, images, img_width, img_height in tqdm(iter(test_loader)):
        images = [img.to(device) for img in images]

        with torch.no_grad():
            outputs = model(images)

        for idx, output in enumerate(outputs):
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            # Denormalize the bounding boxes
            for box, score in zip(boxes, scores):
                if score >= threshold:
                    exist = 1
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2, score = box_denormalize(x1, y1, x2, y2, img_width[idx], img_height[idx], score)
                    results.append({
                        "confidence": score,
                        "point1_x": x1, "point1_y": y1,
                        "point3_x": x2, "point3_y": y2,
                    })

    if results:
        results_df = pd.DataFrame(results)
        denormalized_boxes = [
            [x1, y1, x2, y2, confidence]
            for (x1, y1, x2, y2, confidence) in zip(results_df["point1_x"], results_df["point1_y"], results_df["point3_x"], results_df["point3_y"], results_df["confidence"]) 
        ]
    else:
        denormalized_boxes = None
    
    return denormalized_boxes



def fasterrcnn_inference(img_path,model_path):
    
    test_dataset = CustomDataset(img_path,transforms=get_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model=build_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    res_plotted = Image.open(img_path).convert("RGB")
    res_plotted=res_plotted.resize((640,640), Image.Resampling.BILINEAR)
    boxes=inference(model, test_loader, device, threshold=0.8)
    
    if (boxes == None):
        disease_name = "정상"
        confidence = None
    
    else:
        disease_name = "결석"
        confidence = sum([i[-1] for i in boxes])/len(boxes)
        res_plotted = Image.open(img_path).convert("RGB")
        #
        draw = ImageDraw.Draw(res_plotted)
        
        for box in boxes:
            x1, y1, x2, y2, score = box
            print( x1, y1, x2, y2, score )
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=4)
            disease_name="결석"
            res_plotted = add_text_to_image(res_plotted, f"{disease_name} {score:.2f}", position=(x1, y1-55), font_size=50)
    
    
    return res_plotted, disease_name, confidence
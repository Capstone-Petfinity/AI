from ultralytics import YOLO
from PIL import Image
import cv2 
import matplotlib.pyplot as plt

# Load a model
model = YOLO('./best.pt')
path="./D_62_20150709_CM_0032_ABN_Mu05_20211217_1430.jpg"

img = Image.open(path)
results = model(img)
res_plotted = results[0].plot() # plot() 함수를 이용해서 이미지 내에 bounding box나 mask 등의 result 결과를 그릴 수 O

# plt.figure(figsize=(12, 12))
# plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)) 
# plt.show()

Image.save(res_plotted,'./result.jpg')
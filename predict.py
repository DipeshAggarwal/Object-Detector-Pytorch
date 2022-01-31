from core import config
from core.helper import info
from torchvision import transforms
import mimetypes
import argparse
import immutils
import pickle
import torch
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default="output/text_paths.txt")
args = vars(ap.parse_args())

image_paths = open(args["input"]).read().strip().split("\n")

info("Loading Object Detector")
model = torch.load(config.MODEL_PATH).to(config.DEVICE)
model.eval()
le = pickle.loads(open(config.LE_PATH, "rb").read())

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

for image_path in image_paths:
    image = cv2.imread(image_path)
    orig = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))
    
    image = torch.from_numpy(image)
    image = transforms(image).to(config.DEVICE)
    image = image.unsqueeze(0)
    
    box_preds, label_preds = model(image)
    start_x, start_y, end_x, end_y = box_preds[0]
    
    label_preds = torch.nn.Softmax(dim=-1)(label_preds)
    i = label_preds.argmax(dim=-1).cpu()
    label = le.inverse_transform(i)[0]
    
    orig = imutils.resize(orig, width=600)
    h, w = orig.shape[:2]
    
    start_x = int(start_x * w)
    start_y = int(start_x * h)
    end_x - int(end_x * w)
    end_x - int(end_x * h)

    y =  start_y - 10 if start_y - 10 > 10 else start_y + 10
    cv2.putText(orig, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(orig, (startX, startY), (end_x, end_y), (0, 255, 0), 2)
    
    cv2.imshow("Output", orig)
    cv2.waitKey(0)

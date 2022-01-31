from core.custom_tensor_dataset import CustomTensorDataset
from core.bbox_regressor import ObjectDetector
from core.helper import info
from core import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import time
import cv2
import os

info("Loading dataset")
data = []
labels = []
bboxes = []
image_paths = []

for csv_path in paths.list_files(config.ANNOTS_PATH, validExts=(".csv")):
    rows = open(csv_path).read().strip().split("\n")
    
    for row in rows:
        row = row.split(",")
        filename, start_x, start_y, end_x, end_y, label = row
        
        image_path = os.path.sep.join([config.IMAGES_PATH, label, filename])
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        start_x = float(start_x) / w
        start_y = float(start_y) / h
        end_x = float(end_x) / w
        end_y = float(end_y) / h
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        data.append(image)
        labels.append(label)
        bboxes.append((start_x, start_y, end_x, end_y))
        image_paths.append(image_path)

# Convert to NumPy Array        
data = np.array(data, dtype="float32")
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
image_paths = np.array(image_paths)

le = LabelEncoder()
labels = le.fit_transform(labels)

split = train_test_split(data, labels, bboxes, image_paths, test_size=0.2, random_state=42)

train_images, test_images = split[:2]
train_labels, test_labels = split[2:4]
train_bboxes, test_bboxes = split[4:6]
train_paths, test_paths = split[6:]

# Convert to Pytorch Tensor
train_images, test_images = torch.tensor(train_images), torch.tensor(test_images)
train_labels, test_labels = torch.tensor(train_labels), torch.tensor(test_labels)
train_bboxes, test_bboxes = torch.tensor(train_bboxes), torch.tensor(test_bboxes)

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

train_ds = CustomTensorDataset((train_images, train_labels, train_bboxes), transforms=transforms)
test_ds = CustomTensorDataset((test_images, test_labels, test_bboxes), transforms=transforms)

info("Total Training Samples: {}".format(len(train_ds)))
info("Total Test Samples: {}".format(len(test_ds)))

train_steps = len(train_ds) // config.BATCH_SIZE
test_steps = len(test_ds) // config.BATCH_SIZE

train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)
test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)

info("Saving Testing Image Paths")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(test_paths))
f.close()

resnet = resnet50(pretrained=True)

# Freeze all model layers
for param in resnet.parameters():
    param.requires_grad = False
    
obj_detector = ObjectDetector(resnet, len(le.classes_))
obj_detector = obj_detector.to(config.DEVICE)

class_loss_func = CrossEntropyLoss()
bbox_loss_func = MSELoss()

opt = Adam(obj_detector.parameters(), lr=config.INIT_LR)

H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [], "val_class_acc": []}

info("Training the Network")
start_time = time.time()

for e in tqdm(range(config.NUM_EPOCHS)):
    obj_detector.train()
    
    total_train_loss = 0
    total_val_loss = 0
    
    train_correct = 0
    val_correct = 0
    
    for images, labels, bboxes in train_loader:
        images, labels, bboxes = images.to(config.DEVICE), labels.to(config.DEVICE), bboxes.to(config.DEVICE)
        
        predictions = obj_detector(images)
        bbox_loss = bbox_loss_func(predictions[0], bboxes)
        class_loss = class_loss_func(predictions[1], labels)
        total_loss = (config.BBOX * bbox_loss) + (config.LABELS * class_loss)
        
        # Zero out the gradient
        opt.zero_grad()
        # Perform back propogation
        total_loss.backward()
        opt.step()
        
        total_train_loss += total_loss
        train_correct += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()

    with torch.no_grad():
        obj_detector.eval()
        
        for images, labels, bboxes in test_loader:
            images, labels, bboxes = images.to(config.DEVICE), labels.to(config.DEVICE), bboxes.to(config.DEVICE)
            
            predictions = obj_detector(images)
            bbox_loss = bbox_loss_func(predictions[0], bboxes)
            class_loss = class_loss_func(predictions[1], labels)
            total_loss = (config.BBOX * bbox_loss) + (config.LABELS * class_loss)
            
            total_val_loss += total_loss
            val_correct = (predictions[1].argmax(1) == labels).type(torch.float).sum().item()


    avg_train_loss = total_train_loss / train_steps
    avg_val_loss = total_val_loss / test_steps  

    train_correct = train_correct / len(train_ds)
    val_correct = val_correct / len(test_ds)
    
    H["total_train_loss"].append(avg_train_loss.cpu().detach().numpy())
    H["train_class_acc"].append(train_correct)
    H["total_val_loss"].append(avg_val_loss.cpu().detach().numpy())
    H["val_class_acc"].append(val_correct)
    
    info("Epoch: {}/{}".format(e+1, config.NUM_EPOCHS))
    print("Train Loss: {:.6f}, Train Accuracy: {:.4f}".format(avg_train_loss, train_correct))
    print("Val Loss: {:.6f}, Val Accuracy: {:.4f}".format(avg_val_loss, val_correct))

end_time = time.time()
info("Total time to train the model: {:.2f}s".format(end_time-start_time))

info("Saving Object Detector Model")
torch.save(obj_detector, config.MODEL_PATH)

info("Saving Label encoder")
f = open(config.LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()

plt.style.use("ggplot")
plt.figure()
plt.plot(H["total_train_loss"], label="Total Train Loss")
plt.plot(H["total_val_loss"], label="Total Val Loss")
plt.plot(H["train_class_acc"], label="Train Class Acc")
plt.plot(H["val_class_acc"], label="Val Class Acc")
plt.title("Total Training Loss and Classification Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

plotPath = os.path.sep.join([config.PLOTS_PATH, "training.png"])
plt.savefig(plotPath)

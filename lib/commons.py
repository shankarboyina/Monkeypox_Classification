import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Existing ResNet18 functionality remains unchanged

label_map = {
    0: "Chickenpox",
    1: "Measles",
    2: "Monkeypox",
    3: "Normal"
}
classes = ('Chickenpox', 'Measles', 'Monkeypox', 'Normal')
PATH = 'models/resnet18_net.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def load_model():
    """Load the ResNet18 model."""
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(classes))
    model.to(device)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()
    return model

def image_loader(image_name):
    """load image, returns cuda tensor"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    picture = Image.open(image_name)
    image = data_transform(picture)
    images=image.reshape(1,1,64,64)
    new_images = images.repeat(1, 3, 1, 1)
    return new_images

def predict(model, image_name):
    '''

    pass the model and image url to the function
    Returns: a list of pox types with decreasing probability
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    picture = Image.open(image_name)
    image = data_transform(picture)
    images=image.reshape(1,1,64,64)
    new_images = images.repeat(1, 3, 1, 1)

    outputs=model(new_images)

    _, predicted = torch.max(outputs, 1)
    ranked_labels=torch.argsort(outputs,1)[0]
    probable_classes=[]
    for label in ranked_labels:
        probable_classes.append(classes[label.numpy()])
    probable_classes.reverse()
    return probable_classes

# YOLOv8 Integration
def load_yolov8_model():
    """Load the YOLOv8 model."""
    yolov8_model = YOLO('models\last.pt')  # Update with the actual path
    return yolov8_model

def predict_yolov8(model, image_file):
    """Make predictions using YOLOv8."""
    image = Image.open(image_file)
    image = np.array(image)
    results = model(image)
    detections = results[0]
    class_names = detections.names
    confidences = detections.probs.data.tolist()
    detected_class = class_names[np.argmax(confidences)]
    return detected_class, confidences

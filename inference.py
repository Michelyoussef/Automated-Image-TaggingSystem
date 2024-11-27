import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

def model_fn(model_dir):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 80)  
    model.load_state_dict(torch.load(f"{model_dir}/model.pth"))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data)
    return torch.sigmoid(output)

def output_fn(prediction, accept):
    if accept == 'application/json':
        threshold = 0.5
        predicted_tags = (prediction > threshold).nonzero(as_tuple=True)[1].tolist()
        return json.dumps(predicted_tags)
    raise ValueError("Unsupported accept type: {}".format(accept))
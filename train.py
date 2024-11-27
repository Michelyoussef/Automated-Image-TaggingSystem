import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import boto3
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pandas._libs.internals":
            return getattr(pd._libs.internals, name)
        return super().find_class(module, name)

def load_pickle_safely(file_path):
    try:
        with open(file_path, 'rb') as f:
            return CustomUnpickler(f).load()
    except Exception as e:
        print(f"Error loading pickled data: {e}")
        return None

class COCODataset(Dataset):
    def __init__(self, csv_file, bucket_name, image_prefix, transform=None):
        self.data = pd.read_csv(csv_file)
        self.bucket_name = bucket_name
        self.image_prefix = image_prefix
        self.transform = transform
        self.s3 = boto3.client('s3')
        print(f"Loaded CSV file from {csv_file}")
        print(f"Number of images: {len(self.data)}")
        print(f"First few rows of CSV:")
        print(self.data.head())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['file_name']
        s3_key = f"{self.image_prefix}/images/{img_name}" 

        try:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            image = Image.open(io.BytesIO(obj['Body'].read())).convert('RGB')
        except Exception as e:
            print(f"Error loading image {s3_key}: {e}")
            return None

        tags = torch.tensor(eval(self.data.iloc[idx]['tags']), dtype=torch.float32)
        tags = torch.clamp(tags, 0, 1)
        if self.transform:
            image = self.transform(image)

        return image, tags

def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None

    images, tags = zip(*batch)

    
    max_len = max(len(t) for t in tags)
    padded_tags = [torch.cat([t, torch.zeros(max_len - len(t))]) for t in tags]

    images = torch.stack(images)
    tags = torch.stack(padded_tags)

    return images, tags

def train(args):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Fixed size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    dataset = COCODataset(
        csv_file=os.path.join(args.data_dir, 'image_data.csv'),
        bucket_name='sagemaker-eu-north-1-495599732227',
        image_prefix='preprocessed2',  
        transform=transform
    )

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)

    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 80)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        all_targets = []
        all_predictions = []
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                print(f"Skipping empty batch {batch_idx}")
                continue

            data, target = batch
            data, target = data.to(device), target.to(device)

            try:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target.float())
                loss.backward()
                optimizer.step()

                
                output = torch.sigmoid(output)  
                predicted = (output > 0.5).float()  
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                running_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(
                        f'Epoch {epoch + 1}/{args.epochs} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}')
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Data shape: {data.shape}, Target shape: {target.shape}")
                continue
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='micro')
        recall = recall_score(all_targets, all_predictions, average='micro')
        f1 = f1_score(all_targets, all_predictions, average='micro')

        
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f} - "
              f"Precision: {precision:.4f} - Recall: {recall:.4f} - F1-score: {f1:.4f}")
    
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args = parser.parse_args()
    train(args)
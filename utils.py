import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os

def load_data(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {
        'train': datasets.ImageFolder(f'{data_dir}/train', transform=data_transforms['train']),
        'validation': datasets.ImageFolder(f'{data_dir}/valid', transform=data_transforms['validation']),
        'test': datasets.ImageFolder(f'{data_dir}/test', transform=data_transforms['test']),
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
        'validation': DataLoader(image_datasets['validation'], batch_size=32, shuffle=False, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4),
    }
    
    return dataloaders, image_datasets['train'].class_to_idx

def save_checkpoint(model, optimizer, epochs, save_dir, class_to_idx):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'class_to_idx': class_to_idx,
        'classifier': model.classifier
    }
    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg13(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    image = Image.open(image_path)

    image.thumbnail((256, 256))

    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = (left + 224) 
    bottom = (top + 224) 
    image = image.crop((left, top, right, bottom))

    np_image = np.array(image) / 255
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds

    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image).float()

def predict(image, model, top_k=5):
    image = process_image(image_path)
    
    image = torch.from_numpy(image).type(torch.FloatTensor)
    
    image = image.unsqueeze(0)
    
    model = model.to('cuda')
    image = image.to('cuda')
    
    model.eval()
    
    with torch.no_grad():
        output = model(image)
    
    probs = torch.exp(output)
    
    top_probs, top_indices = probs.topk(topk)

    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    return top_probs, top_classes

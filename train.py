import argparse
import torch
from torch import nn, optim
from torchvision import models
import os
from utils import load_data, save_checkpoint  # Assume utils.py handles loading data and saving checkpoints

def get_input_args():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg13', help='Model architecture (e.g., "vgg13")')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    return parser.parse_args()

def train_model(model, criterion, optimizer, dataloaders, num_epochs=5):
    model = model.to('cuda') 
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

def main():
    # Get input arguments
    args = get_input_args()

    # Load data
    dataloaders, class_to_idx = load_data(args.data_dir)

    # Load a pre-trained model
    model = models.__dict__[args.arch](pretrained=True)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    input_size = model.classifier[0].in_features
    # Define the classifier
    model.classifier = nn.Sequential(
        nn.Linear(input_size, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model = train_model(model, criterion, optimizer, dataloaders, num_epochs=args.epochs)
    
    # # Save the checkpoint
    save_checkpoint(model, optimizer, args.epochs, args.save_dir, class_to_idx)

if __name__ == '__main__':
    main()

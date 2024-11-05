import argparse
import torch
from torchvision import models
from PIL import Image
import json
from utils import load_checkpoint, process_image, predict  # Assume utils.py handles loading checkpoints, processing images, and making predictions

def get_input_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name.')
    
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category names json file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    return parser.parse_args()

def predict(image_path, model, top_k=5):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    model.eval()
    
    # Process the image
    image = process_image(image_path)
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Move to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        
    # Get the top K probabilities and corresponding classes
    top_probs, top_indices = ps.topk(top_k, dim=1)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # Convert indices to classes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_indices]
    
    return top_probs, top_classes


def main():
    # Get input arguments
    args = get_input_args()

    # Load the model
    model = load_checkpoint(args.checkpoint)
    
    # Use GPU if specified
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Make predictions
    probs, classes = predict(args.image_path, model, args.top_k)
    flower_names = [cat_to_name[class_] for class_ in classes]    
    # Print results
    print(f"Probabilities: {probs}")
    print(f"Classes: {classes}")
    print(f"Flower names: {flower_names}")

if __name__ == '__main__':
    main()

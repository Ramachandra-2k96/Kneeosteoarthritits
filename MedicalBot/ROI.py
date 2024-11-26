import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import matplotlib.colors as mcolors
import io
import base64

class ModelVisualizer:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Get the last convolutional layer
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Conv2d):
                self.target_layer = module
                break
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_image: torch.Tensor, target_class: int):
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        output[0, target_class].backward()
        
        # Generate weights
        weights = torch.mean(self.gradients, dim=(2, 3))[0]
        
        # Generate cam
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]
        
        cam = torch.maximum(cam, torch.tensor(0, device=self.device))
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()

def visualize_areas_of_interest(model: nn.Module, image_path: str, device: torch.device):
    # Load and preprocess image
    original_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(original_image).unsqueeze(0).to(device)
    
    # Get prediction
    visualizer = ModelVisualizer(model, device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax().item()
        prediction_probabilities = torch.softmax(output, dim=1)[0]
    
    # Generate and process heatmap
    heatmap = visualizer.generate_heatmap(input_tensor, predicted_class)
    heatmap_resized = Image.fromarray(heatmap).resize(original_image.size, Image.LANCZOS)
    heatmap_resized = np.array(heatmap_resized)
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    plt.imshow(heatmap_resized, 
              cmap=mcolors.LinearSegmentedColormap.from_list('custom', [(0,0,0,0), 'yellow', 'red'], N=10),
              alpha=0.5)
    plt.axis('off')
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return {
        'image_base64': image_base64,
        'predicted_class': predicted_class,
        'prediction_probabilities': prediction_probabilities.cpu().numpy().tolist()
    }

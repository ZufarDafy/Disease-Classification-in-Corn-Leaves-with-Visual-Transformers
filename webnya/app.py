from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import torch
from vit_pytorch import ViT

app = Flask(__name__)

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 4,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
)
# Load the trained model
model_path = 'D://PENELITIAN//PENELITIAN - DIASH//vit_new_train1.pth'
model = v
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device)
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Define the classes
classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy'] # replace with your actual class names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Load the image
        image = Image.open(file)

        # Transform the image
        input_image = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(input_image)
            _, predicted_class = torch.max(output, 1)

        # Get predicted class
        predicted_label = classes[predicted_class.item()]

        return jsonify({'predicted_class': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
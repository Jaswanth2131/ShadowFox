import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

def load_and_prep_image(filename, img_shape=32):
    """
    Reads an image from filename, turns it into a tensor
    and reshapes it to (img_shape, img_shape, colour_channel).
    """
    img = Image.open(filename)
    img = img.resize((img_shape, img_shape))
    img = np.array(img) / 255.0
    
    # Handle RGBA images (drop alpha channel)
    if img.shape[-1] == 4:
        img = img[..., :3]
        
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model_path, image_path):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading image from {image_path}...")
    img = load_and_prep_image(image_path)
    
    print("Predicting...")
    pred = model.predict(img)
    pred_class_index = np.argmax(pred)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    pred_class = class_names[pred_class_index]
    confidence = np.max(pred)
    
    print(f"Prediction: {pred_class} (Confidence: {confidence:.2f})")
    return pred_class, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict image class using trained model.')
    parser.add_argument('image', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default='model.h5', help='Path to the trained model file')
    
    args = parser.parse_args()
    predict_image(args.model, args.image)

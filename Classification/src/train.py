import os
import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from model import create_model
import numpy as np

def main():
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Class names for reference
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Create model
    print("Creating model...")
    model = create_model()

    # Train model
    print("Starting training...")
    history = model.fit(train_images, train_labels, epochs=2, 
                        validation_data=(test_images, test_labels))

    # Save model
    model_path = 'model.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f"Test accuracy: {test_acc}")

if __name__ == "__main__":
    main()

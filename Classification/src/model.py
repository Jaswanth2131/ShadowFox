import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Creates a simple CNN model for image classification.
    
    Args:
        input_shape: tuple, shape of input images (height, width, channels).
        num_classes: int, number of output classes.
        
    Returns:
        model: compiled tf.keras.Model.
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()

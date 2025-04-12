import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def load_data():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    return train_images, train_labels, test_images, test_labels

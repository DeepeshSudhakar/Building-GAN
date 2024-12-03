import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Build Autoencoder Model
def create_autoencoder(encoding_dim=32):
    # Encoder
    input_img = keras.layers.Input(shape=(28, 28, 1))
    
    # Convolutional Encoder
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Flatten and dense layer for bottleneck
    x = keras.layers.Flatten()(x)
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(x)
    
    # Decoder
    x = keras.layers.Dense(7*7*64, activation='relu')(encoded)
    x = keras.layers.Reshape((7, 7, 64))(x)
    
    # Convolutional Decoder
    x = keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Create autoencoder model
    autoencoder = keras.Model(input_img, decoded)
    encoder = keras.Model(input_img, encoded)
    
    return autoencoder, encoder

# Create and compile the autoencoder
autoencoder, encoder = create_autoencoder(encoding_dim=32)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
history = autoencoder.fit(x_train, x_train, 
                          epochs=50, 
                          batch_size=128, 
                          shuffle=True, 
                          validation_data=(x_test, x_test))

# Visualize Reconstruction
def visualize_reconstruction(autoencoder, x_test):
    # Predict reconstructions
    decoded_imgs = autoencoder.predict(x_test[:10])
    
    # Plot original and reconstructed images
    plt.figure(figsize=(20, 4))
    for i in range(10):
        # Display original
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Display reconstruction
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.tight_layout()
    plt.savefig('autoencoder_reconstruction.png')
    plt.close()

# Visualize Latent Space
def visualize_latent_space(encoder, x_test, y_test):
    # Encode test images
    encoded_imgs = encoder.predict(x_test)
    
    # Plot 2D latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], 
                          c=y_test, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('2D Latent Space Representation')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.tight_layout()
    plt.savefig('latent_space_visualization.png')
    plt.close()

# Loss Visualization
def plot_training_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.close()

# Perform visualizations
visualize_reconstruction(autoencoder, x_test)
visualize_latent_space(encoder, x_test, y_test)
plot_training_loss(history)

# Print model summary
autoencoder.summary()

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess MNIST dataset
(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]

# Generator Model
def build_generator():
    model = keras.Sequential([
        keras.layers.Dense(7*7*256, input_dim=100),
        keras.layers.Reshape((7, 7, 256)),
        keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same'), 
        keras.layers.Activation('tanh')
    ])
    return model

# Discriminator Model
def build_discriminator():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
                             input_shape=(28, 28, 1)),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Combined GAN Model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = keras.Sequential([generator, discriminator])
    return model

# Training parameters
BATCH_SIZE = 256
EPOCHS = 10000
NOISE_DIM = 100

# Build models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5),
                      loss='binary_crossentropy', metrics=['accuracy'])
gan = build_gan(generator, discriminator)
gan.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy')

# Training loop
def train_gan(epochs, batch_size=256):
    # Create figure to save generated images
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]
        
        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))
        fake_imgs = generator.predict(noise)
        
        # Prepare labels
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
        
        # Generate and save images every 1000 epochs
        if epoch % 1000 == 0:
            noise = np.random.normal(0, 1, (16, NOISE_DIM))
            gen_imgs = generator.predict(noise)
            
            # Rescale images to [0,1]
            gen_imgs = 0.5 * gen_imgs + 0.5
            
            # Plot generated images
            for i in range(4):
                for j in range(4):
                    axs[i,j].imshow(gen_imgs[i*4+j].reshape(28,28), cmap='gray')
                    axs[i,j].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'gan_generated_digits_{epoch}.png')
            plt.close()

# Train the GAN
train_gan(EPOCHS, BATCH_SIZE)

# Final image generation
noise = np.random.normal(0, 1, (16, NOISE_DIM))
gen_imgs = generator.predict(noise)
gen_imgs = 0.5 * gen_imgs + 0.5

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(gen_imgs[i].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.savefig('final_gan_generated_digits.png')
plt.close()

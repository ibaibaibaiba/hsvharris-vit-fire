# !pip install tensorflow-cpu==2.10
# !pip install pip install tensorflow-directml-plugin
# !pip install Pillow
# !pip install scikit-learn
# conda install matplotlib

# Import the necessary libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.layers.experimental import preprocessing

import pickle
import random

import os
import datetime
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import sys


# to do - belum diubah buat ngasih bounding box dll. / buat main lain


# Define the image size and number of channels
image_size = 256
num_channels = 3
batch_size = 32

# data augmentation
# Load your dataset
def load_dataset(data_dir,image_size=image_size, batch_size=batch_size, seed=42):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        seed=seed
    )
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        seed=seed
    )

    return train_generator, validation_generator


# Preprocessing function for the images GAK DIPAKE
def preprocess_image(image, label):
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# Multi-head self-attention layer
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_rate = dropout
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training):
        batch_size = tf.shape(inputs)[0]
        query = self.separate_heads(self.query_dense(inputs), batch_size)
        key = self.separate_heads(self.key_dense(inputs), batch_size)
        value = self.separate_heads(self.value_dense(inputs), batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

# Stochastic Depth REMOVED

# Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.dropout1 = layers.Dropout(dropout)
        
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.keras.activations.gelu),
            layers.Dense(embed_dim),
        ])
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        norm_inputs = self.layernorm1(inputs)
        attn_output = self.att(norm_inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        
        norm_out1 = self.layernorm2(out1)
        mlp_output = self.mlp(norm_out1)
        mlp_output = self.dropout2(mlp_output, training=training)
        return out1 + mlp_output


# Vision Transformer
class VisionTransformer(tf.keras.Model):
    def __init__(self, image_size, patch_size, num_layers, d_model, num_heads, mlp_dim, channels=3, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.pos_encoding = self.add_weight("pos_encoding", initializer="zeros", shape=(1, self.num_patches + 1, d_model))

        self.patch_proj = layers.Dense(d_model)
        self.cls_token = self.add_weight("cls_token", initializer="zeros", shape=(1, 1, d_model))

        self.transformer_layers = [TransformerBlock(d_model, num_heads, mlp_dim, dropout) for _ in range(num_layers)]

        self.mlp_head = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.keras.activations.relu), # using ReLU
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid")  # Single output neuron with a sigmoid activation function
        ])


    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')
        patches = tf.reshape(patches, (batch_size, self.num_patches, self.patch_dim))
        return patches

    def call(self, inputs, training):
        batch_size = tf.shape(inputs)[0]
        patches = self.extract_patches(inputs)
        patches_proj = self.patch_proj(patches)

        cls_tokens = tf.broadcast_to(self.cls_token, (batch_size, 1, self.d_model))
        x = tf.concat([cls_tokens, patches_proj], axis=1)
        x += self.pos_encoding

        for layer in self.transformer_layers:
            x = layer(x, training)

        cls_token_final = x[:, 0]
        logits = self.mlp_head(cls_token_final)
        return logits

def warmup_scheduler(epoch, lr, warmup_epochs=10, initial_lr=0.0001, target_lr=0.001):
    if epoch < warmup_epochs:
        return initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
    else:
        return lr
    


lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(warmup_scheduler)
initial_learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)


# MAIN

# Define the model parameters
# COBA EKSPERIMEN NILAI LAIN
patch_size = 16 #16
num_layers = 8 #8
d_model = 64 #64
num_heads = 4 #4
mlp_dim = 128 #128

# Instantiate the Vision Transformer model
model = VisionTransformer(image_size, patch_size, num_layers, d_model=d_model, num_heads=num_heads, mlp_dim=mlp_dim, channels=3, dropout=0.1)

# Compile the model
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# learning rate default 1e-4

# Define the training parameters
batch_size = 32
epochs = 100
#========
################# LOKASI DATASET ######################
# data_directory = ".\Dataset\pure"
#data_directory = ".\Dataset\edited\\rgb"
# data_directory = ".\Dataset\edited\hsv"
#data_directory = ".\\Dataset\\final\\objects\\rgb"
#data_directory = ".\\Dataset\\final\\objects\\hsv"
#data_directory = ".\\Dataset\\final\\objectsEdit\\rgb"
#data_directory = ".\\Dataset\\final\\objectsEdit\\hsv"
#data_directory = ".\\Dataset\\final\\xmlroi"
data_directory = ".\\Dataset\\final\\xmlroiPlusobjects\\rgb"
#data_directory = ".\\Dataset\\final\\xmlroiPlusobjects\\hsv"
#images, labels = load_dataset(data_directory)
train_generator, validation_generator = load_dataset(data_directory)

#
# Define the checkpoint path and file name
checkpoint_filepath = '.\\Saved Model\\checkpoint\\'

# Create a ModelCheckpoint callback that saves the best model based on validation accuracy
# model_checkpoint_callback = ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=False,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True,
#     verbose=1
# )
class SaveBestEpoch(tf.keras.callbacks.Callback):
    def __init__(self, filepath, model_filepath):
        super().__init__()
        self.best_val_accuracy = 0
        self.best_epoch_stats = {}
        self.filepath = filepath
        self.model_filepath = model_filepath

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_accuracy = logs.get('val_accuracy', 0)
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.best_epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': logs.get('loss', np.nan),
                'train_accuracy': logs.get('accuracy', np.nan),
                'val_loss': logs.get('val_loss', np.nan),
                'val_accuracy': logs.get('val_accuracy', np.nan)
            }
            with open(self.filepath, 'w') as f:
                f.write(str(self.best_epoch_stats))
            
            # Save the best model
            self.model.save(self.model_filepath)

stats_filepath = '.\\Saved Model\\final_model\\stats.txt'
model_filepath = '.\\Saved Model\\final_model'

save_best_epoch_callback = SaveBestEpoch(filepath=stats_filepath, model_filepath=model_filepath)


# Train the model with the callback
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[save_best_epoch_callback, lr_schedule_callback]
)
#

# MODEL NTAR DI OUTPUT
with open('.\\Saved Model\\final_model\\history_data.pkl', 'wb') as f:
    pickle.dump(history.history, f)

with open('.\\Saved Model\\final_model\\history_data.pkl', 'rb') as f:
    history_data = pickle.load(f)

# Evaluate the model on the test dataset
#loss, accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)

#print(f"Test loss: {loss:.4f}")
#print(f"Test accuracy: {accuracy:.4f}")

# plt.figure(figsize=(10, 5))
# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.title("Training and Validation Loss")
# #plt.show()

# # Plot the training and validation accuracy
# plt.figure(figsize=(10, 5))
# plt.plot(history.history["accuracy"], label="Training Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.title("Training and Validation Accuracy")
# plt.show()

def plot_history(history_data):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_data['loss'], label='Training Loss')
    plt.plot(history_data['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_data['accuracy'], label='Training Accuracy')
    plt.plot(history_data['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

plot_history(history_data)

# save model
now = datetime.datetime.now()
datetime = now.strftime("%Y-%m-%d %H.%M.%S")
print(os.path.join(".\\Saved Model\\",datetime))
#model.save(os.path.join(".\\Saved Model\\",datetime))
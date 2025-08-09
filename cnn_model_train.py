import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Image dimensions
img_height, img_width = 128, 128
batch_size = 16
epochs = 10

# Dataset directory
base_dir = "dataset"
model_path = "model/cnn_model.h5"

# Ensure the model output directory exists
os.makedirs("model", exist_ok=True)

# Preprocess data and define training/validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training data loader
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation data loader
val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes: faulty and quality_ok
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# Save trained model
model.save(model_path)
print(f"✅ Model saved to: {model_path}")

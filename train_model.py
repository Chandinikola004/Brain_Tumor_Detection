import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

# Load and preprocess data
def load_data(data_dir):
    images = []
    labels = []
    for label in ["yes", "no"]:
        path = os.path.join(data_dir, label)
        class_num = 1 if label == "yes" else 0
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))
            images.append(image)
            labels.append(class_num)
    images = np.array(images).reshape(-1, 128, 128, 1)
    labels = np.array(labels)
    return images, labels

# Directory containing the 'yes' and 'no' folders
data_dir = "C:\\Users\\chand\\OneDrive\\Desktop\\brain_tumor_detection\\augmented_data"
images, labels = load_data(data_dir)

# Normalize images
images = images / 255.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the CNN model
input_shape = (128, 128, 1)
model_input = Input(shape=input_shape)

x = Conv2D(32, (3, 3), padding='same')(model_input)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(128)(x)
x = Activation('relu')(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=model_input, outputs=x)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save("C:\\Users\\chand\\OneDrive\\Desktop\\brain_tumor_detection\\brain_tumor_detection_model.h5")

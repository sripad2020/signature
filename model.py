import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.applications import ResNet50
from keras.api.layers import Input, TimeDistributed, LSTM, Dense, GlobalAveragePooling2D
from keras.api.models import Model
from keras.api.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from PIL import UnidentifiedImageError

# Settings
IMG_SIZE = (224, 224)
SEQUENCE_LENGTH = 1
BATCH_SIZE = 8
EPOCHS = 10
DATASET_PATH = 'signatures'

# 1. Model
def build_model():
    base_model = ResNet50(include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), weights='imagenet')
    base_model.trainable = False
    input_tensor = Input(shape=(SEQUENCE_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3))
    features = TimeDistributed(base_model)(input_tensor)
    features = TimeDistributed(GlobalAveragePooling2D())(features)
    x = LSTM(128, return_sequences=False)(features)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 2. Safe Preprocessing
def preprocess_image(img_path):
    try:
        img = load_img(img_path, target_size=IMG_SIZE)
        img = img_to_array(img) / 255.0
        return np.expand_dims(np.expand_dims(img, axis=0), axis=0)  # shape: (1, 1, H, W, 3)
    except (UnidentifiedImageError, OSError) as e:
        print(f"[Skipping] Could not process {img_path}: {e}")
        return None

# 3. Dataset Loader with Error Handling
def load_dataset(real_path, forged_path):
    X, y = [], []
    for label, folder in zip([0, 1], [real_path, forged_path]):
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            processed = preprocess_image(path)
            if processed is not None:
                X.append(processed)
                y.append(label)
    return np.vstack(X), np.array(y)


real_dir = os.path.join(DATASET_PATH, 'full_org')
forged_dir = os.path.join(DATASET_PATH, 'full_forg')
X, y = load_dataset(real_dir, forged_dir)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = build_model()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)


model.save("signature_verifier.h5")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()


sample_real = preprocess_image("signatures/full_org/original_1_1.png")
sample_forged = preprocess_image("signatures/full_forg/forgeries_1_1.png")

if sample_real is not None:
    print("Prediction (Real Signature):", model.predict(sample_real))

if sample_forged is not None:
    print("Prediction (Forged Signature):", model.predict(sample_forged))
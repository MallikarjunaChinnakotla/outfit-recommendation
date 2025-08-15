import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB3

# -------------------------------
# 1. Load CSV and Check Images
# -------------------------------
df = pd.read_csv("your csv file path")
df = df[df['outfit_image'].apply(lambda path: os.path.exists(path))]
print(f"✅ Valid images found: {len(df)}")

# -------------------------------
# 2. Encode Labels
# -------------------------------
df['label'] = df['outfit_type']
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_enc'], random_state=42)
# 3. Data Augmentation & Loaders

IMG_SIZE = 299
BATCH_SIZE = 32

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

def load_image(path, label, augment=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = preprocess_input(image)
    if augment:
        image = data_augmentation(image)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((train_df['outfit_image'], train_df['label_enc']))
train_ds = train_ds.map(lambda x, y: load_image(x, y, augment=True), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_df['outfit_image'], val_df['label_enc']))
val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 4. Handle Class Imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label_enc']),
    y=train_df['label_enc']
)
class_weights_dict = dict(enumerate(class_weights))


# 5. EfficientNetB3 Model

base_model = EfficientNetB3(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet")
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)


checkpoint_path = "/content/drive/MyDrive/best_efficientnetb3.keras"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 7. Initial Training

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weights_dict
)

# 8. Fine-Tuning

print("🔓 Fine-tuning top layers...")
base_model.trainable = True
for layer in base_model.layers[:-40]:  # Freeze all but top 40 layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weights_dict
)

# 9. Save Final Model Again (optional)

model.save("/file path locationto store /final_efficientnetb3.keras")
print("✅ Model saved successfully to Google Drive.")

# 10. Final Evaluation
loss, acc = model.evaluate(val_ds)
print(f"✅ Final Accuracy: {round(acc * 100, 2)}%")

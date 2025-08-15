import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Trained Model
# -------------------------------
model = tf.keras.models.load_model("/content/drive/MyDrive/final_efficientnetb3.keras")
print("✅ Model loaded!")

# -------------------------------
# 2. Load Dataset & Encode Labels
# -------------------------------
df = pd.read_csv("/content/drive/MyDrive/projectc_modified_drive_paths.csv")
df = df[df['outfit_image'].apply(lambda path: os.path.exists(path))]

le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['outfit_type'])

# -------------------------------
# 3. Take Inputs
# -------------------------------
gender = input("Enter Gender (male/female): ").strip().lower()
skin_tone = input("Enter Skin Tone (light/medium/dark): ").strip().lower()
outfit_type = input(f"Enter Outfit Type {list(le.classes_)}: ").strip().lower()

# -------------------------------
# 4. Preprocess Function
# -------------------------------
IMG_SIZE = 299

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image

# -------------------------------
# 5. Filter Matching Images
# -------------------------------
filtered_df = df[
    (df['gender'].str.lower() == gender) &
    (df['skin_tone'].str.lower() == skin_tone) &
    (df['outfit_type'].str.lower() == outfit_type)
]

if filtered_df.empty:
    print("❌ No images found matching these inputs.")
    exit()

# -------------------------------
# 6. Predict and Display Top Images
# -------------------------------
X = tf.stack([preprocess_image(path) for path in filtered_df['outfit_image']])
predictions = model.predict(X)

target_label = le.transform([outfit_type])[0]
scores = predictions[:, target_label]

top_indices = np.argsort(scores)[-10:][::-1]  # Top 5 scores
top_rows = filtered_df.iloc[top_indices]

print(f"✅ Top {len(top_rows)} images for gender={gender}, skin_tone={skin_tone}, outfit_type={outfit_type}")

for _, row in top_rows.iterrows():
    img = tf.io.decode_jpeg(tf.io.read_file(row['outfit_image']))
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis("off")
    plt.title(outfit_type)
    plt.show()


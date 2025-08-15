import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# 1. App Config
st.set_page_config(page_title="Strike & Style", layout="wide")

# 2. Load Model and Dataset

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("final_efficientnetb3 (1).keras")
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("projectc_modified_final_with_folder.csv")

    # Normalize paths if coming from Google Drive
    df['outfit_image'] = df['outfit_image'].apply(
        lambda x: x.split("/content/drive/MyDrive/")[-1] if "/content/drive/MyDrive/" in x else x
    )

    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['outfit_type'])
    return df, le

model = load_model()
df, le = load_data()


# 3. UI Inputs

st.title("👗 Strike & Style")

col1, col2, col3 = st.columns(3)
gender = col1.selectbox("Select Gender", options=df['gender'].str.lower().unique(), index=1)
skin_tone = col2.selectbox("Select Skin Tone", options=df['skin_tone'].str.lower().unique(), index=1)

filtered_options = df[
    (df['gender'].str.lower() == gender.lower()) &
    (df['skin_tone'].str.lower() == skin_tone.lower())
]['outfit_type'].unique()

outfit_type = col3.selectbox("Select Outfit Type", options=filtered_options)


# 4. Image Preprocessing

IMG_SIZE = 299

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image


# 5. Run Prediction

if st.button("Get Strike & Style", use_container_width=True):
    with st.spinner("Finding the best strike & styles for you..."):
        # Filter based on selected options
        filtered_df = df[
            (df['gender'].str.lower() == gender.lower()) &
            (df['skin_tone'].str.lower() == skin_tone.lower()) &
            (df['outfit_type'].str.lower() == outfit_type.lower())
        ]

        # Remove duplicates by image path
        filtered_df = filtered_df.drop_duplicates(subset='outfit_image').reset_index(drop=True)

        if filtered_df.empty:
            st.error("No matching strike & style found. Try different filters.")
        else:
            # Preprocess images
            image_paths = filtered_df['outfit_image'].tolist()
            try:
                X = tf.stack([preprocess_image(path) for path in image_paths])
            except Exception as e:
                st.error(f"Error processing images: {e}")
                st.stop()

            # Make predictions
            predictions = model.predict(X)
            target_label = le.transform([outfit_type])[0]
            scores = predictions[:, target_label]

            # Sort by highest scores and get top N
            top_indices = np.argsort(scores)[-30:][::-1]  # top 30, descending
            top_rows = filtered_df.iloc[top_indices].drop_duplicates(subset='outfit_image').head(15).reset_index(drop=True)
            top_scores = scores[top_indices][:len(top_rows)]

            st.success(f"Showing top {len(top_rows)} matching strike & styles for you!")

            # Display the top outfit images with prediction scores
            cols = st.columns(5)
            for idx, row in top_rows.iterrows():
                try:
                    img = Image.open(row['outfit_image'])
                    score = top_scores[idx]
                    with cols[idx % 5]:
                        st.image(img, caption=f"{outfit_type} ({score:.2f})", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load image: {row['outfit_image']} — {str(e)}")


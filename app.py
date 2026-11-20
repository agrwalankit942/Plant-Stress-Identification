
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Plant Stress Identification",
    page_icon="🌿",
    layout="wide"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f7faf7;
}

.title {
    font-size: 36px;
    font-weight: bold;
    color: #166534;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
}

.prediction-box {
    background-color: #dcfce7;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "Plant_Stress_Results"
)

ALEXNET_PATH = os.path.join(
    MODEL_DIR,
    "AlexNet.keras"
)

VGG16_PATH = os.path.join(
    MODEL_DIR,
    "VGG16.keras"
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "class_names.json"
)


# =====================================================
# 4. LOAD CLASS NAMES
# =====================================================

@st.cache_data
def load_classes():

    with open(CLASS_PATH, "r") as file:
        return json.load(file)


# =====================================================
# 5. LOAD TRAINED MODELS
# =====================================================

@st.cache_resource
def load_alexnet():

    return tf.keras.models.load_model(
        ALEXNET_PATH,
        compile=False
    )


@st.cache_resource
def load_vgg16():

    return tf.keras.models.load_model(
        VGG16_PATH,
        compile=False
    )


# =====================================================
# 6. IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image, model_type):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    if model_type == "AlexNet":

        image_array = image_array / 255.0

    elif model_type == "VGG16":

        image_array = preprocess_input(
            image_array
        )

    return image_array


# =====================================================
# 7. PREDICTION FUNCTION
# =====================================================

def predict_image(image, model, model_type, class_names):

    processed_image = preprocess_image(
        image,
        model_type
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    if len(predictions) != len(class_names):
        raise ValueError(
            "Model output does not match the number of classes."
        )

    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence = float(
        predictions[predicted_index] * 100
    )

    return predicted_class, confidence, predictions


# =====================================================
# 8. APPLICATION HEADER
# =====================================================

st.markdown(
    '<div class="title">🌿 Plant Stress Identification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Based Paddy Leaf Disease Classification'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# 9. SIDEBAR
# =====================================================

st.sidebar.title("🌱 Project Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Disease Prediction",
        "Model Comparison",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    "**CNN Architectures**"
)

st.sidebar.write("1. AlexNet")
st.sidebar.write("2. VGG16")

st.sidebar.divider()

st.sidebar.caption(
    "Plant Stress Identification | Deep Learning Project"
)


# =====================================================
# 10. HOME PAGE
# =====================================================

if page == "Home":

    st.header("Welcome to Paddy Leaf Stress Detection")

    st.write(
        """
        This application uses Convolutional Neural Networks
        to classify paddy leaf images into five categories.
        Upload a leaf image and use the trained models to
        obtain a prediction.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("CNN Models", "2")

    col2.metric("Leaf Categories", "5")

    col3.metric("Image Size", "224 × 224")

    st.subheader("Supported Categories")

    categories = [
        ("Blast", "🔥"),
        ("BLB", "🍂"),
        ("Healthy", "🌿"),
        ("Hispa", "🐛"),
        ("Leaf Spot", "🟤")
    ]

    cols = st.columns(5)

    for col, (name, emoji) in zip(cols, categories):

        with col:

            st.markdown(f"### {emoji}")

            st.write(name)

    st.info(
        "Go to Disease Prediction to upload a paddy leaf image."
    )


# =====================================================
# 11. DISEASE PREDICTION PAGE
# =====================================================

elif page == "Disease Prediction":

    st.header("Paddy Leaf Disease Prediction")

    model_choice = st.selectbox(
        "Select CNN Model",
        ["AlexNet", "VGG16"]
    )

    uploaded_file = st.file_uploader(
        "Upload Paddy Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Paddy Leaf Image")

        st.image(
            image,
            caption="Input Paddy Leaf Image",
            width=400
        )

        if st.button(
            "Predict Disease",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner("Analyzing paddy leaf image..."):

                    class_names = load_classes()

                    if model_choice == "AlexNet":

                        model = load_alexnet()

                    else:

                        model = load_vgg16()

                    predicted_class, confidence, probabilities = (
                        predict_image(
                            image,
                            model,
                            model_choice,
                            class_names
                        )
                    )

                st.success("Prediction Completed!")

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        '<div class="prediction-box">'
                        '<h3>Predicted Category</h3>'
                        f'<h2>{predicted_class}</h2>'
                        '</div>',
                        unsafe_allow_html=True
                    )

                with col2:

                    st.metric(
                        "Confidence Score",
                        f"{confidence:.2f}%"
                    )

                # Probability table

                results = pd.DataFrame({

                    "Class": class_names,

                    "Probability (%)": probabilities * 100

                })

                results = results.sort_values(
                    "Probability (%)",
                    ascending=False
                )

                st.subheader("Class-wise Prediction")

                st.dataframe(
                    results.style.format({
                        "Probability (%)": "{:.2f}%"
                    }),
                    use_container_width=True
                )

                # Probability chart

                st.subheader("Prediction Probability Chart")

                chart_data = results.set_index("Class")

                st.bar_chart(
                    chart_data["Probability (%)"]
                )

            except Exception as e:

                st.error(f"Prediction failed: {e}")


# =====================================================
# 12. MODEL COMPARISON PAGE
# =====================================================

elif page == "Model Comparison":

    st.header("AlexNet vs VGG16 Comparison")

    st.write(
        "Upload an image to compare predictions from both models."
    )

    uploaded_file = st.file_uploader(
        "Upload Paddy Leaf Image",
        type=["jpg", "jpeg", "png"],
        key="comparison_upload"
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Selected Paddy Leaf",
            width=400
        )

        if st.button(
            "Compare Both Models",
            type="primary"
        ):

            try:

                with st.spinner("Comparing AlexNet and VGG16..."):

                    class_names = load_classes()

                    alexnet = load_alexnet()

                    vgg16 = load_vgg16()

                    alex_class, alex_conf, alex_prob = predict_image(
                        image,
                        alexnet,
                        "AlexNet",
                        class_names
                    )

                    vgg_class, vgg_conf, vgg_prob = predict_image(
                        image,
                        vgg16,
                        "VGG16",
                        class_names
                    )

                comparison = pd.DataFrame({

                    "Model": ["AlexNet", "VGG16"],

                    "Predicted Class": [
                        alex_class,
                        vgg_class
                    ],

                    "Confidence (%)": [
                        alex_conf,
                        vgg_conf
                    ]

                })

                st.subheader("Prediction Comparison")

                st.dataframe(
                    comparison.style.format({
                        "Confidence (%)": "{:.2f}%"
                    }),
                    use_container_width=True
                )

                st.subheader("Class Probability Comparison")

                probability_df = pd.DataFrame({

                    "Class": class_names,

                    "AlexNet (%)": alex_prob * 100,

                    "VGG16 (%)": vgg_prob * 100

                })

                st.dataframe(
                    probability_df.style.format({
                        "AlexNet (%)": "{:.2f}%",
                        "VGG16 (%)": "{:.2f}%"
                    }),
                    use_container_width=True
                )

                st.bar_chart(
                    probability_df.set_index("Class")
                )

                if alex_class == vgg_class:

                    st.success(
                        f"Both models predicted: {alex_class}"
                    )

                else:

                    st.warning(
                        "The two models produced different predictions."
                    )

            except Exception as e:

                st.error(f"Comparison failed: {e}")


# =====================================================
# 13. ABOUT PROJECT PAGE
# =====================================================

elif page == "About Project":

    st.header("About Plant Stress Identification")

    st.write(
        """
        The project aims to classify paddy leaf stress using
        deep learning and image classification techniques.
        """
    )

    st.subheader("Models Used")

    st.markdown("""
    **AlexNet**
    - Baseline CNN architecture.
    - Learns image features through convolution layers.
    - Uses fully connected classification layers.

    **VGG16**
    - Deep convolutional neural network.
    - Uses pretrained ImageNet weights.
    - Extracts deep image features through transfer learning.
    """)

    st.subheader("Classification Categories")

    for category in [
        "Blast",
        "BLB",
        "Healthy",
        "Hispa",
        "Leaf Spot"
    ]:

        st.write("•", category)

    st.subheader("Evaluation Metrics")

    st.write("""
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Computational Training Time
    """)


# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <center>
    <p style="color:gray;">
    Plant Stress Identification | Deep Learning Project
    <br>
    Developed using Python, TensorFlow and Streamlit
    </p>
    </center>
    """,
    unsafe_allow_html=True
)

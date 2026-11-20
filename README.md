# 🌿 Plant Stress Identification – Paddy Leaf Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-green" />
  <img src="https://img.shields.io/badge/Models-AlexNet%20%7C%20VGG16-purple" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

## 📌 Project Overview

**Plant Stress Identification** is a Deep Learning-based image classification project designed to identify diseases and stress conditions in paddy leaves using Convolutional Neural Networks (CNN).

The project implements two CNN architectures, **AlexNet and VGG16**, to classify paddy leaf images into five different categories.

A user-friendly Streamlit web application allows users to upload leaf images, predict the category, visualize prediction probabilities, and compare the results of both models.

## 🎯 Objectives

- Identify paddy leaf diseases and stress conditions using deep learning.
- Implement and compare AlexNet and VGG16 architectures.
- Apply image preprocessing and data augmentation.
- Evaluate model performance using standard classification metrics.
- Develop an interactive Streamlit-based prediction application.

## 🌱 Classification Categories

The system supports five paddy leaf categories:

| Category | Description |
|---|---|
| Blast | Rice blast disease |
| BLB | Bacterial Leaf Blight |
| Healthy | Healthy paddy leaves |
| Hispa | Rice hispa damage |
| Leaf Spot | Paddy leaf spot disease |

## 🧠 Deep Learning Models

### 1. AlexNet

- Baseline CNN architecture.
- Learns image features through convolutional layers.
- Uses ReLU activation and max pooling.
- Includes fully connected classification layers.
- Trained from scratch on the paddy leaf dataset.

### 2. VGG16

- Deep convolutional neural network.
- Uses pretrained ImageNet weights.
- Applies transfer learning for feature extraction.
- Uses Global Average Pooling.
- Includes a custom classification layer for five classes.

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| NumPy | Numerical operations |
| Pandas | Data processing |
| Matplotlib | Visualization |
| Seaborn | Confusion matrix visualization |
| Scikit-learn | Model evaluation |
| Streamlit | Web application |
| Pillow | Image processing |

## 🔄 Project Workflow

1. Dataset Collection
2. Data Loading
3. Image Preprocessing
4. Image Resizing (224 × 224)
5. Data Augmentation
6. AlexNet Model Development
7. VGG16 Transfer Learning
8. Model Training and Validation
9. Model Evaluation
10. Performance Comparison
11. New Image Prediction
12. Streamlit Application Development

## 📊 Model Evaluation Metrics

Both models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Computational Training Time

### Model Performance Comparison

The following metrics are generated from the actual testing results.

| Metric | AlexNet | VGG16 |
|---|---|---|
| Accuracy | Run experiment | Run experiment |
| Precision | Run experiment | Run experiment |
| Recall | Run experiment | Run experiment |
| F1-score | Run experiment | Run experiment |
| Training Time | Run experiment | Run experiment |

*Replace these placeholders with your actual experimental results.*

## 🚀 Streamlit Web Application

The project includes an interactive web application with the following modules:

### 🏠 Home
- Project introduction.
- Supported classification categories.
- Overview of CNN models.

### 🔍 Disease Prediction
- Upload paddy leaf images.
- Select AlexNet or VGG16.
- Display predicted category.
- Display confidence percentage.
- Visualize class-wise prediction probabilities.

### 📈 Model Comparison
- Compare AlexNet and VGG16 predictions.
- Compare confidence scores.
- Visualize class-wise probabilities.

### ℹ️ About Project
- Project description.
- CNN architecture details.
- Classification categories.
- Evaluation metrics.

## 📂 Project Structure

```text
Plant-Stress-Identification/
│
├── Dataset/
│   ├── train/
│   │   ├── Blast/
│   │   ├── BLB/
│   │   ├── healthy/
│   │   ├── hispa/
│   │   └── leaf_spot/
│   │
│   ├── val/
│   └── test/
│
├── Plant_Stress_Results/
│   ├── AlexNet.keras
│   ├── VGG16.keras
│   └── class_names.json
│
├── app.py
├── requirements.txt
└── README.md
```

## 💻 Installation and Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Plant-Stress-Identification.git
```

### Step 2: Navigate to Project

```bash
cd Plant-Stress-Identification
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run Streamlit Application

```bash
streamlit run app.py
```

### Step 6: Open in Browser

```text
http://localhost:8501
```

## 📦 Requirements

The project uses the following Python libraries:

```text
streamlit
tensorflow==2.21.0
numpy
pandas
matplotlib
pillow
```



## 🔮 Future Enhancements

- Add more paddy leaf disease categories.
- Implement Grad-CAM for model explainability.
- Develop real-time leaf disease detection.
- Add disease management recommendations verified by agricultural experts.
- Deploy the application using Streamlit Community Cloud.
- Improve prediction performance using fine-tuning and additional datasets.

## 🎓 Project Information

**Project Title:** Plant Stress Identification (Paddy Leaves)

**Domain:** Artificial Intelligence and Deep Learning

**Models:** AlexNet and VGG16

**Application:** Streamlit Web Application

**Task:** Multi-class Image Classification

## 👨‍💻 Author

**Your Name:** Ankit Agrawal

**Course:** Master of Computer Applications (MCA)

**Specialization:** Artificial Intelligence and Machine Learning

---

<p align="center">
  <b>🌿 Plant Stress Identification Using Deep Learning</b>
  <br>
  Developed with Python, TensorFlow, CNN and Streamlit.
</p>

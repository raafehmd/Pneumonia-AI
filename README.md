# 🧠 Pneumonia Detection using AI and Logical Programming

A powerful AI-driven healthcare tool that detects pneumonia from chest X-ray images using a convolutional neural network (CNN) built with TensorFlow, integrated with a Prolog-based logic system for simple medical advice.

<div align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" />
  <img src="https://img.shields.io/badge/framework-streamlit-orange" />
  <img src="https://img.shields.io/badge/status-active-brightgreen" />
</div>

---

## 💡 Project Description

This is a university project that explores multiple programming paradigms:

-  🧮 **Procedural** (Data preprocessing with `ImageDataGenerator`)
-  🧱 **Object-Oriented** (CNN model wrapped in a Python class)
-  🧠 **Logical** (Medical recommendations using Prolog)

It trains a model and uses it to analyze lung X-ray images and determine the probability of pneumonia, and then provides rule-based guidance using logical programming.

---

## 📦 Features

-  Upload chest X-ray images for instant prediction
-  Intuitive web interface using Streamlit
-  Real-time model inference
-  Simple Prolog-based medical guidance
-  Designed to be easy to run and extend

---

## 📁 Dataset Used

[Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 🚀 Getting Started

### Prerequisites

Make sure you have **Python 3.10+** installed.

> 📌 Optional: [Install SWI-Prolog](https://www.swi-prolog.org/Download.html) if you want to run the logic engine.

### 1. Clone the repository

```
git clone https://github.com/your-username/pneumonia-ai.git
cd pneumonia-ai
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```
python -m venv venv
# Activate it:
venv\Scripts\activate     # On Windows
# OR
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Requirements

```
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```
streamlit run app.py
```

## 📸 How to Use

-  Launch the app using the command above.
-  Your browser will open a local Streamlit web interface.
-  Upload one or more lung X-ray images (JPG or PNG format).
-  Click "Predict".
-  The app will:
-  Analyze each image with a pretrained model
-  Display the probability of pneumonia
-  Use Prolog rules to offer logical medical advice

## 📦 Technologies Used

-  Category Tools/Frameworks
-  Language Python, Prolog
-  Machine Learning TensorFlow / Keras
-  GUI/Web Interface Streamlit
-  Image Processing Keras ImageDataGenerator
-  Logic Programming SWI-Prolog
-  Utilities NumPy, scikit-learn, PIL

## 🗂️ Project Structure

```
📦 pneumonia-ai
├── src/
│ ├── preprocessing.py
│ ├── model.py
│ ├── evaluation.py
│ ├── predict_single.py
│ ├── prolog_interface.py
│ ├── app.py
│ └── rules.pl
├── saved_model/
│ └── pneumonia_model.h5
├── data/
│ └── chest_xray/ (dataset directory)
├── requirements.txt
├── README.md
```

## 📊 Sample Output

Image Name | Probability | Prediction | Logical Advice
person1_virus_6.jpeg | 0.89 | Pneumonia | Seek further medical evaluation.
person2_bacteria.jpeg | 0.12 | Normal | Continue routine health monitoring.

## 📈 Future Improvements

-  Support for additional image formats and resolutions
-  Advanced GUI with multiple views and image previews
-  REST API for integration with external systems
-  Better logic engine for more complex diagnoses

## 🧪 For Developers

If you wish to retrain or fine-tune the model:

-  Download the dataset from Kaggle and place it under /data/chest_xray/.
-  Use main.py in the src/ folder to train a new model.

Save the model using:

```
model.save("saved_model/pneumonia_model.h5")
```

-  The Streamlit app will automatically use the saved model.

## 📝 License

This project is licensed for academic and educational purposes only.

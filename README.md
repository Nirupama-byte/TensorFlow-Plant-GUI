# 🍅 AI-Powered Tomato Disease Detector

This project is a desktop application that uses a deep learning model to identify 10 different tomato plant diseases. The app is built with Python, using a Convolutional Neural Network (CNN) trained with TensorFlow/Keras and a modern user interface built with CustomTkinter.

[App Screenshot](Screenshot 2025-10-19 002720.png)


---

### ✨ Features
* **Modern GUI:** A clean and user-friendly desktop interface built with CustomTkinter.
* **Real-Time Prediction:** Load an image of a tomato leaf and get an instant diagnosis.
* **High Accuracy:** The model is a fine-tuned MobileNetV2, trained on the PlantVillage dataset to identify 10 classes (9 diseases + 1 healthy).
* **Self-Contained:** The trained model (`.h5` file) is included in this repository.

---

### 🛠 Tech Stack
* **Python 3**
* **TensorFlow / Keras:** For building and running the deep learning model.
* **CustomTkinter:** For the modern desktop GUI.
* **Pillow (PIL):** For loading and processing images in the app.
* **Numpy:** For numerical operations.

---

### 🚀 How to Run

1.  **Clone or Download the Repository:**
    ```bash
    git clone (https://github.com/Nirupama-byte/TensorFlow-Plant-GUI.git)
  

2.  **Install the Required Libraries:**
    Make sure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    The trained model (`tomato_model_v2_robust.h5`) is already included. Just run the Python script:
    ```bash
    python app_gui.py
    ```

---

### 🔬 Model & Performance
The model was intentionally trained to be robust. It uses aggressive data augmentation (brightness, zoom, rotation) and fine-tuning to better identify "in-the-wild" images, not just "textbook" examples from the dataset. This helps solve the common "domain shift" problem in machine learning.

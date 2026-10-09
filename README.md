# Handwriting Character Recognition Using CNN

An end-to-end handwritten character recognition application using:
- EMNIST Balanced dataset
- TensorFlow/Keras CNN
- OpenCV preprocessing
- Streamlit interface

## Project structure

```text
Handwriting-Recognition-CNN/
├── app.py
├── train_model.py
├── evaluate_model.py
├── preprocessing.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── models/
└── outputs/
```

## 1. Create environment

Recommended: Python 3.10 or 3.11.

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## 3. Train

```powershell
python train_model.py
```

The first run downloads EMNIST Balanced automatically through TensorFlow Datasets.

The trained model is saved as:

```text
models/handwriting_cnn.keras
```

## 4. Evaluate

```powershell
python evaluate_model.py
```

This generates:

```text
outputs/confusion_matrix.png
outputs/training_accuracy.png
outputs/training_loss.png
```

## 5. Run Streamlit

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

## Dataset

EMNIST Balanced contains 47 classes of handwritten digits and letters in 28x28 grayscale format.

## Important

The Streamlit app requires a trained model. Run `train_model.py` before starting the app.

## Future extension

The current system recognizes one character at a time. It can later be extended with character segmentation and sequence recognition to convert complete handwritten words/sentences into digital text.

# 🚗 Car Model Identification Using Deep Learning

This project uses a Convolutional Neural Network (CNN) to classify car images into specific models based on the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). The model is trained using PyTorch and evaluated with multiple classification metrics.

---

## 📂 Dataset

The dataset used is downloaded from Kaggle:
- **Name:** Stanford Car Dataset by Classes Folder
- **Link:** https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder

The dataset is split into train and test folders, each containing subfolders representing car model classes.

---

## 🔧 Technologies and Libraries

- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn

---

## 🧠 Model Architecture

The model is based on a pretrained convolutional neural network (e.g., ResNet18). Transfer learning is used to adapt it for fine-grained car classification.

---

## 🔁 Training

Key steps in training:
- Image transformations (resize, normalization, augmentation)
- CrossEntropyLoss
- Adam optimizer
- GPU acceleration (if available)

---

## 📊 Results

- **Accuracy:** XX%
- **Precision:** XX%
- **Recall:** XX%
- Classification report and confusion matrix are generated.
- Sample training and validation loss curves are plotted.

*(Note: Replace `XX%` with your actual results)*

---

## 📈 Visualizations

Include sample outputs like:
- Training & validation loss curve
- Accuracy over epochs
- Confusion matrix (optional)
- Sample predictions (optional)

---

## ▶️ Usage

```bash
# Clone the repository
git clone https://github.com/hoomanfallah97/car-model-identification.git

# Run the notebook
Open 'car model identification.ipynb' in Jupyter or Google Colab.

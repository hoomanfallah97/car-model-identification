# 🚗 Car Model Identification Using Deep Learning

This project uses a Convolutional Neural Network (CNN) to classify car images into specific models based on the [Stanford Cars Dataset](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder). The model is trained using PyTorch and evaluated with multiple classification metrics.

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

The model is based on ResNet-50, a deep convolutional neural network renowned for its performance in image classification tasks. By leveraging transfer learning, the pretrained ResNet-50 model is fine-tuned to classify car images into specific models.

---

## 🔁 Training

Key steps in training:
- Image transformations (resize, normalization, augmentation)
- CrossEntropyLoss
- Adam optimizer
- GPU acceleration (if available)

---

## 📊 Results

- **Accuracy:** 93.00%
- **Precision:** 0.93
- **Recall:** 0.93
- Classification report and confusion matrix are generated.
- Sample training and validation loss curves are plotted.


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

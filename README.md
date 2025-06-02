# Plant Disease Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A deep learning project for automated detection and classification of plant diseases using Convolutional Neural Networks (CNN). This system can identify 38 different plant diseases across multiple crop types with high accuracy.

## 🌱 Project Overview

Plant diseases are a major threat to agricultural productivity and food security worldwide. Early detection and accurate identification of plant diseases can help farmers take timely action to minimize crop losses. This project leverages deep learning techniques to automatically classify plant diseases from leaf images.

### Key Features
- **38 Disease Classes**: Covers multiple crops including Tomato, Apple, Corn, Potato, Grape, and more
- **High Accuracy**: Achieves ~97% training accuracy and ~94.5% validation accuracy
- **CNN Architecture**: Custom deep learning model optimized for plant disease classification
- **Easy to Use**: Simple inference pipeline for single image predictions
- **Comprehensive Dataset**: Trained on 70,295+ images with proper validation split

## 📊 Dataset

The project uses the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle.

### Dataset Statistics
- **Training Images**: 70,295 files
- **Validation Images**: 17,572 files
- **Total Classes**: 38 plant disease categories
- **Image Format**: RGB JPG images
- **Input Size**: 128x128 pixels

### Supported Plant Types & Diseases

#### 🍅 Tomato (10 classes)
- Yellow Leaf Curl Virus, Mosaic Virus, Target Spot, Spider Mites, Septoria Leaf Spot, Late Blight, Early Blight, Bacterial Spot, Leaf Mold, Healthy

#### 🍎 Apple (4 classes)
- Cedar Apple Rust, Black Rot, Apple Scab, Healthy

#### 🌽 Corn (4 classes)
- Northern Leaf Blight, Common Rust, Gray Leaf Spot, Healthy

#### 🥔 Potato (3 classes)
- Early Blight, Late Blight, Healthy

#### 🍇 Grape (4 classes)
- Black Rot, Esca (Black Measles), Leaf Blight, Healthy

#### Other Crops
- Bell Pepper, Cherry, Peach, Orange, Soybean, Squash, Strawberry, and Raspberry

## 🏗️ Model Architecture

The model uses a Convolutional Neural Network (CNN) with the following architecture:

```
Input Layer (128, 128, 3)
    ↓
Conv2D (32 filters, 3x3) + ReLU + MaxPooling2D
    ↓
Conv2D (64 filters, 3x3) + ReLU + MaxPooling2D + Dropout(0.25)
    ↓
Conv2D (128 filters, 3x3) + ReLU + MaxPooling2D
    ↓
Conv2D (256 filters, 3x3) + ReLU + MaxPooling2D + Dropout(0.4)
    ↓
Conv2D (512 filters, 3x3) + ReLU + MaxPooling2D
    ↓
Flatten + Dense(512, ReLU) + Dropout(0.4)
    ↓
Dense(38, Softmax) - Output Layer
```

**Total Parameters**: 7,842,762

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Plant_Disease_Detection.git
cd Plant_Disease_Detection
```

2. **Install dependencies**
```bash
pip install -r requirement.txt
```

### Required Packages
- tensorflow==2.10.0
- scikit-learn==1.3.0
- numpy==1.24.3
- matplotlib==3.7.2
- seaborn==0.13.0
- pandas==2.1.0
- streamlit

## 📁 Project Structure

```
Plant_Disease_Detection/
├── README.md
├── requirement.txt
├── Train_plant_disease.ipynb      # Model training notebook
├── Test_plant_disease.ipynb       # Model testing notebook
├── trained_plant_disease_model.keras  # Saved trained model
├── train/                         # Training dataset
│   ├── class1/
│   ├── class2/
│   └── ...
├── valid/                         # Validation dataset
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/                          # Test images
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## 💻 Usage

### Training the Model

1. **Prepare your dataset**: Ensure your training and validation data are in the `train/` and `valid/` directories respectively.

2. **Run the training notebook**:
```bash
jupyter notebook Train_plant_disease.ipynb
```

3. **Execute all cells** to train the model. The trained model will be saved as `trained_plant_disease_model.keras`.

### Testing/Inference

1. **Load the testing notebook**:
```bash
jupyter notebook Test_plant_disease.ipynb
```

2. **Run inference** on new images by placing them in the `test/` directory and executing the notebook cells.

### Single Image Prediction

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Load and preprocess image
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction[0])
confidence = np.max(prediction[0]) * 100

print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

## 📈 Results

### Training Performance
- **Training Accuracy**: ~97%
- **Validation Accuracy**: ~94.5%
- **Training Epochs**: 10
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32

### Model Performance
- High accuracy across diverse plant disease categories
- Robust performance on validation data
- Efficient inference for real-time applications

## 🔬 Technical Details

### Data Preprocessing
- Images resized to 128×128 pixels
- RGB color mode
- Normalization applied (pixel values scaled to [0,1])
- Data augmentation techniques for better generalization

### Training Configuration
- **Optimizer**: Adam with learning rate 0.0001
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Regularization**: Dropout layers (0.25, 0.4)
- **Validation Split**: Separate validation dataset

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) for providing the comprehensive plant disease dataset
- TensorFlow team for the robust deep learning framework
- The open-source community for valuable resources and tools

## 📞 Contact

For questions, suggestions, or collaborations, please feel free to reach out:

- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)
- **Email**: your.email@example.com

---

**Made with ❤️ for sustainable agriculture and food security** 
# 🌿 Plant Disease Prediction Project

A deep learning-based system for detecting and diagnosing plant diseases from images of leaves and plants using Convolutional Neural Networks (CNN) and Transfer Learning.

## 📋 Overview

This project uses state-of-the-art deep learning models to classify plant diseases from images. It supports multiple pre-trained architectures including MobileNetV2, ResNet50, EfficientNetB0, and custom CNN models.

### Features

- ✅ Multiple pre-trained model architectures (Transfer Learning)
- ✅ Custom CNN architecture option
- ✅ Data augmentation for robust training
- ✅ Easy-to-use web interface with Streamlit
- ✅ Batch prediction capability
- ✅ Model checkpointing and early stopping
- ✅ Training visualization and metrics
- ✅ Support for multiple plant species and diseases

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Plant disease prediction project"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
Plant disease prediction project/
├── data/
│   ├── raw/                    # Original dataset (organized by class)
│   └── processed/              # Processed data
├── models/
│   └── saved_models/           # Trained models
├── src/
│   ├── data_loader.py          # Data loading and preprocessing
│   └── model.py                # Model architectures
├── notebooks/                  # Jupyter notebooks for experiments
├── config.py                   # Configuration settings
├── train.py                    # Training script
├── predict.py                  # Prediction script
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📊 Dataset Preparation

### Option 1: Use PlantVillage Dataset

Download the PlantVillage dataset (or any plant disease dataset) and organize it as follows:

```
data/raw/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Apple___Black_rot/
│   ├── image1.jpg
│   └── ...
├── Tomato___healthy/
│   └── ...
└── ...
```

### Option 2: Use Your Own Dataset

Organize your images in subdirectories, where each subdirectory name is the class name:

```
data/raw/
├── disease_class_1/
│   ├── img1.jpg
│   └── img2.jpg
├── disease_class_2/
└── healthy/
```

**Popular Datasets:**
- PlantVillage Dataset: [Kaggle Link](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- Plant Pathology Dataset: [Kaggle Link](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8)

## 🎓 Training the Model

### Basic Training

```bash
python train.py --data_dir data/raw --epochs 50
```

### Advanced Training Options

```bash
python train.py \
    --data_dir data/raw \
    --model_type MobileNetV2 \
    --epochs 100 \
    --batch_size 32
```

**Available model types:**
- `MobileNetV2` (default, fast and accurate)
- `ResNet50` (higher accuracy, more memory)
- `EfficientNetB0` (excellent accuracy/speed tradeoff)
- `VGG16` (classic architecture)
- `Custom` (custom CNN from scratch)

### Configuration

Modify `config.py` to adjust:
- Image size
- Batch size
- Learning rate
- Data augmentation parameters
- Model architecture
- And more...

## 🔮 Making Predictions

### Single Image Prediction

```bash
python predict.py \
    --model_path models/saved_models/plant_disease_MobileNetV2_20231217_120000/best_model.keras \
    --image_path path/to/your/image.jpg \
    --top_k 5
```

### Batch Prediction

```bash
python predict.py \
    --model_path models/saved_models/plant_disease_MobileNetV2_20231217_120000/best_model.keras \
    --image_dir path/to/image/directory \
    --output_dir predictions
```

### Visualize Predictions

```bash
python predict.py \
    --model_path models/saved_models/plant_disease_MobileNetV2_20231217_120000/best_model.keras \
    --image_path path/to/your/image.jpg \
    --visualize \
    --output_dir visualizations
```

## 🌐 Web Application

Launch the interactive web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Web App Features:
- 📤 Upload plant images
- 🔍 Instant disease prediction
- 📊 Confidence scores for multiple diseases
- 📥 Download prediction results
- 🎨 Beautiful, user-friendly interface

## 🛠️ Usage Examples

### Python API

```python
from predict import PlantDiseasePredictor

# Initialize predictor
predictor = PlantDiseasePredictor(
    model_path='models/saved_models/your_model/best_model.keras',
    class_indices_path='models/saved_models/your_model/class_indices.json'
)

# Predict single image
results = predictor.predict_image('path/to/image.jpg', top_k=3)

print("Top prediction:", results['predictions'][0]['class'])
print("Confidence:", results['predictions'][0]['confidence_percent'])

# Visualize prediction
predictor.visualize_prediction('path/to/image.jpg', save_path='output.png')
```

## 📈 Model Performance

The model tracks the following metrics during training:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Loss**: Categorical cross-entropy loss

Training history plots are automatically saved in the model directory.

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Error:**
   - Reduce batch size in `config.py`
   - Use a smaller model (e.g., MobileNetV2)

2. **Low Accuracy:**
   - Ensure dataset is properly organized
   - Increase number of epochs
   - Try data augmentation
   - Use transfer learning with pre-trained models

3. **Model Not Found:**
   - Check model path is correct
   - Ensure training completed successfully

## 📚 Technical Details

### Model Architecture

The default MobileNetV2 architecture consists of:
- Pre-trained MobileNetV2 base (ImageNet weights)
- Global Average Pooling
- Batch Normalization
- Dropout (0.5)
- Dense layer (512 units, ReLU)
- Output layer (softmax)

### Data Augmentation

Training images are augmented with:
- Rotation (±20 degrees)
- Width/Height shift (20%)
- Horizontal flip
- Zoom (20%)
- Shear transformations

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for additional model architectures
- Mobile app integration
- Real-time video detection
- Multi-language support
- Disease treatment recommendations

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- PlantVillage Dataset
- TensorFlow/Keras team
- Streamlit for the amazing web framework

## 📧 Contact

For questions or suggestions, please open an issue on the project repository.

---

**Happy Plant Disease Detection! 🌿🔬**

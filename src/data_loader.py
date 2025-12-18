"""
Data loader and preprocessing utilities for Plant Disease Prediction
"""
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import config


class PlantDiseaseDataLoader:
    """
    Data loader for plant disease images with augmentation capabilities
    """
    
    def __init__(self, data_dir, image_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE):
        """
        Initialize data loader
        
        Args:
            data_dir: Path to dataset directory
            image_size: Target image size (height, width)
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        
    def create_data_generators(self, validation_split=config.VALIDATION_SPLIT):
        """
        Create training and validation data generators with augmentation
        
        Args:
            validation_split: Fraction of data to use for validation
            
        Returns:
            train_generator, val_generator, class_names
        """
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=config.ROTATION_RANGE,
            width_shift_range=config.WIDTH_SHIFT_RANGE,
            height_shift_range=config.HEIGHT_SHIFT_RANGE,
            shear_range=0.2,
            zoom_range=config.ZOOM_RANGE,
            horizontal_flip=config.HORIZONTAL_FLIP,
            vertical_flip=config.VERTICAL_FLIP,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Validation data (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, val_generator, train_generator.class_indices
    
    def create_data_generators_from_directories(self, train_dir, val_dir):
        """
        Create training and validation data generators from separate directories
        
        Args:
            train_dir: Path to training directory
            val_dir: Path to validation directory
            
        Returns:
            train_generator, val_generator, class_names
        """
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=config.ROTATION_RANGE,
            width_shift_range=config.WIDTH_SHIFT_RANGE,
            height_shift_range=config.HEIGHT_SHIFT_RANGE,
            shear_range=0.2,
            zoom_range=config.ZOOM_RANGE,
            horizontal_flip=config.HORIZONTAL_FLIP,
            vertical_flip=config.VERTICAL_FLIP,
            fill_mode='nearest'
        )
        
        # Validation data (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator, train_generator.class_indices
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.image_size)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def load_dataset_from_directory(self, data_dir=None):
        """
        Load entire dataset from directory structure into memory
        Useful for small to medium datasets
        
        Args:
            data_dir: Path to dataset directory (uses self.data_dir if None)
            
        Returns:
            X_train, X_val, y_train, y_val, class_names
        """
        if data_dir is None:
            data_dir = self.data_dir
            
        images = []
        labels = []
        class_names = sorted(os.listdir(data_dir))
        
        print(f"Loading dataset from {data_dir}")
        print(f"Found {len(class_names)} classes")
        
        for label_idx, class_name in enumerate(class_names):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Loading {len(image_files)} images from {class_name}")
            
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                try:
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.image_size)
                    img = img.astype('float32') / 255.0
                    
                    images.append(img)
                    labels.append(label_idx)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # One-hot encode labels
        y = to_categorical(y, num_classes=len(class_names))
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=config.VALIDATION_SPLIT,
            random_state=42,
            stratify=np.argmax(y, axis=1)
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val, class_names


def get_class_weights(train_generator):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        train_generator: Training data generator
        
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = train_generator.classes
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(classes),
        y=classes
    )
    
    return dict(enumerate(class_weights))

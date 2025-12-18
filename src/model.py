"""
Model architecture for Plant Disease Prediction
Supports multiple pre-trained models and custom CNN
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50, EfficientNetB0, VGG16
)
from tensorflow.keras.optimizers import Adam
import config


class PlantDiseaseModel:
    """
    Plant disease classification model with transfer learning support
    """
    
    def __init__(self, num_classes, model_type=config.MODEL_TYPE, 
                 input_shape=(*config.IMAGE_SIZE, 3)):
        """
        Initialize model architecture
        
        Args:
            num_classes: Number of disease classes
            model_type: Type of model ('MobileNetV2', 'ResNet50', 'EfficientNetB0', 'Custom')
            input_shape: Input image shape (height, width, channels)
        """
        self.num_classes = num_classes
        self.model_type = model_type
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self, freeze_base=config.FREEZE_BASE_LAYERS):
        """
        Build the model architecture
        
        Args:
            freeze_base: Whether to freeze pre-trained base layers
            
        Returns:
            Compiled Keras model
        """
        if self.model_type == 'MobileNetV2':
            self.model = self._build_mobilenet(freeze_base)
        elif self.model_type == 'ResNet50':
            self.model = self._build_resnet(freeze_base)
        elif self.model_type == 'EfficientNetB0':
            self.model = self._build_efficientnet(freeze_base)
        elif self.model_type == 'VGG16':
            self.model = self._build_vgg16(freeze_base)
        elif self.model_type == 'Custom':
            self.model = self._build_custom_cnn()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def _build_mobilenet(self, freeze_base=True):
        """Build MobileNetV2-based model"""
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base layers
        base_model.trainable = not freeze_base
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_resnet(self, freeze_base=True):
        """Build ResNet50-based model"""
        base_model = ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = not freeze_base
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_efficientnet(self, freeze_base=True):
        """Build EfficientNetB0-based model"""
        base_model = EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = not freeze_base
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_vgg16(self, freeze_base=True):
        """Build VGG16-based model"""
        base_model = VGG16(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = not freeze_base
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(512, activation='relu'),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_custom_cnn(self):
        """Build custom CNN from scratch"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classification head
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=config.LEARNING_RATE):
        """
        Compile the model with optimizer and loss
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()]
        )
        
        return self.model
    
    def get_model_summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        return self.model.summary()
    
    def unfreeze_base_layers(self, num_layers=None):
        """
        Unfreeze base layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end. 
                       If None, unfreezes all layers.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if num_layers is None:
            # Unfreeze all layers
            for layer in self.model.layers:
                layer.trainable = True
        else:
            # Unfreeze last num_layers
            for layer in self.model.layers[-num_layers:]:
                layer.trainable = True
        
        print(f"Unfroze layers for fine-tuning")


def create_model(num_classes, model_type=config.MODEL_TYPE):
    """
    Factory function to create and compile a model
    
    Args:
        num_classes: Number of disease classes
        model_type: Type of model architecture
        
    Returns:
        Compiled Keras model
    """
    disease_model = PlantDiseaseModel(num_classes, model_type)
    model = disease_model.build_model()
    model = disease_model.compile_model()
    
    return model

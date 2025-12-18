"""
Training script for Plant Disease Prediction Model
"""
import os
import sys
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from model import PlantDiseaseModel
from data_loader import PlantDiseaseDataLoader, get_class_weights
import config


class PlantDiseaseTrainer:
    """
    Trainer class for plant disease classification model
    """
    
    def __init__(self, data_dir, model_type=config.MODEL_TYPE):
        """
        Initialize trainer
        
        Args:
            data_dir: Path to training data directory
            model_type: Type of model to train
        """
        self.data_dir = data_dir
        self.model_type = model_type
        self.model = None
        self.history = None
        self.class_indices = None
        self.class_weights = None
        self.train_generator = None
        self.val_generator = None
        self.num_classes = None
        
        # Create timestamp for this training run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"plant_disease_{model_type}_{self.timestamp}"
        
    def prepare_data(self):
        """Load and prepare training data"""
        print("=" * 60)
        print("Loading and preparing data...")
        print("=" * 60)
        
        data_loader = PlantDiseaseDataLoader(
            self.data_dir,
            image_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE
        )
        
        # Check if using separate train/val directories
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            # Use separate train/val directories
            print("Using separate train/val directories...")
            self.train_generator, self.val_generator, self.class_indices = \
                data_loader.create_data_generators_from_directories(train_dir, val_dir)
        else:
            # Use single directory with automatic split
            self.train_generator, self.val_generator, self.class_indices = \
                data_loader.create_data_generators(validation_split=config.VALIDATION_SPLIT)
        
        self.num_classes = len(self.class_indices)
        
        print(f"\nNumber of classes: {self.num_classes}")
        print(f"Training samples: {self.train_generator.n}")
        print(f"Validation samples: {self.val_generator.n}")
        print(f"Class indices: {self.class_indices}")
        
        # Calculate class weights for imbalanced data
        self.class_weights = get_class_weights(self.train_generator)
        print(f"\nClass weights: {self.class_weights}")
        
    def build_model(self):
        """Build and compile the model"""
        print("\n" + "=" * 60)
        print(f"Building {self.model_type} model...")
        print("=" * 60)
        
        disease_model = PlantDiseaseModel(
            num_classes=self.num_classes,
            model_type=self.model_type
        )
        
        self.model = disease_model.build_model(freeze_base=config.FREEZE_BASE_LAYERS)
        self.model = disease_model.compile_model(learning_rate=config.LEARNING_RATE)
        
        print("\nModel architecture:")
        self.model.summary()
        
    def setup_callbacks(self):
        """Setup training callbacks"""
        # Create directories
        model_save_dir = os.path.join(config.SAVED_MODELS_DIR, self.model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        
        logs_dir = os.path.join(model_save_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=os.path.join(model_save_dir, 'best_model.keras'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=config.MIN_LR,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=logs_dir,
                histogram_freq=1,
                write_graph=True
            ),
            
            # CSV logging
            CSVLogger(
                filename=os.path.join(model_save_dir, 'training_log.csv'),
                separator=',',
                append=False
            )
        ]
        
        return callbacks, model_save_dir
    
    def train(self, epochs=config.EPOCHS):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
        """
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
        
        callbacks, model_save_dir = self.setup_callbacks()
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(model_save_dir, 'final_model.keras')
        self.model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        # Save class indices
        class_indices_path = os.path.join(model_save_dir, 'class_indices.json')
        with open(class_indices_path, 'w') as f:
            json.dump(self.class_indices, f, indent=4)
        print(f"Class indices saved to: {class_indices_path}")
        
        # Save training configuration
        config_dict = {
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'image_size': config.IMAGE_SIZE,
            'batch_size': config.BATCH_SIZE,
            'epochs': epochs,
            'learning_rate': config.LEARNING_RATE,
            'timestamp': self.timestamp
        }
        config_path = os.path.join(model_save_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"Training configuration saved to: {config_path}")
        
        return self.history, model_save_dir
    
    def plot_training_history(self, save_dir):
        """
        Plot and save training history
        
        Args:
            save_dir: Directory to save plots
        """
        if self.history is None:
            print("No training history available")
            return
        
        history = self.history.history
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0, 0].plot(history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(history['loss'], label='Train Loss')
        axes[0, 1].plot(history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision plot
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Train Precision')
            axes[1, 0].plot(history['val_precision'], label='Val Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall plot
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Train Recall')
            axes[1, 1].plot(history['val_recall'], label='Val Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to: {plot_path}")
        plt.close()
        
    def evaluate(self):
        """Evaluate model on validation set"""
        print("\n" + "=" * 60)
        print("Evaluating model on validation set...")
        print("=" * 60)
        
        results = self.model.evaluate(self.val_generator, verbose=1)
        
        print("\nValidation Results:")
        for metric_name, value in zip(self.model.metrics_names, results):
            print(f"{metric_name}: {value:.4f}")
        
        return results


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Plant Disease Classification Model')
    parser.add_argument('--data_dir', type=str, default=config.RAW_DATA_DIR,
                       help='Path to training data directory')
    parser.add_argument('--model_type', type=str, default=config.MODEL_TYPE,
                       choices=['MobileNetV2', 'ResNet50', 'EfficientNetB0', 'VGG16', 'Custom'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.BATCH_SIZE = args.batch_size
    
    print("\n" + "=" * 60)
    print("PLANT DISEASE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"\nError: Data directory not found: {args.data_dir}")
        print("Please organize your dataset in the following structure:")
        print("  data/raw/")
        print("    ├── class1/")
        print("    │   ├── image1.jpg")
        print("    │   └── image2.jpg")
        print("    ├── class2/")
        print("    │   ├── image1.jpg")
        print("    │   └── image2.jpg")
        return
    
    # Initialize trainer
    trainer = PlantDiseaseTrainer(args.data_dir, args.model_type)
    
    # Prepare data
    trainer.prepare_data()
    
    # Build model
    trainer.build_model()
    
    # Train model
    history, save_dir = trainer.train(epochs=args.epochs)
    
    # Plot training history
    trainer.plot_training_history(save_dir)
    
    # Evaluate model
    trainer.evaluate()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Model saved in: {save_dir}")


if __name__ == "__main__":
    main()

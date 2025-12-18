"""
Utility functions for Plant Disease Prediction
"""
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_classification_report(y_true, y_pred, class_names, save_path=None):
    """
    Generate and save classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the report
        
    Returns:
        Classification report as dictionary
    """
    report = classification_report(y_true, y_pred, 
                                  target_names=class_names,
                                  output_dict=True)
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Classification report saved to: {save_path}")
    
    # Print report
    print("\nClassification Report:")
    print("=" * 80)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return report


def visualize_sample_predictions(model, data_generator, class_names, 
                                num_samples=9, save_path=None):
    """
    Visualize sample predictions
    
    Args:
        model: Trained model
        data_generator: Data generator
        class_names: List of class names
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    # Get a batch of images
    images, labels = next(data_generator)
    predictions = model.predict(images[:num_samples])
    
    # Plot
    n_cols = 3
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Display image
        ax.imshow(images[i])
        
        # Get true and predicted labels
        true_label = class_names[np.argmax(labels[i])]
        pred_label = class_names[np.argmax(predictions[i])]
        confidence = np.max(predictions[i])
        
        # Color based on correctness
        color = 'green' if true_label == pred_label else 'red'
        
        # Title
        ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}',
                    color=color, fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_metrics(history_path, save_dir=None):
    """
    Plot training metrics from saved history
    
    Args:
        history_path: Path to training history CSV
        save_dir: Directory to save plots
    """
    import pandas as pd
    
    history = pd.read_csv(history_path)
    
    metrics = [col for col in history.columns if not col.startswith('val_') 
               and col != 'epoch']
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(history['epoch'], history[metric], label=f'Train {metric}')
        
        val_metric = f'val_{metric}'
        if val_metric in history.columns:
            ax.plot(history['epoch'], history[val_metric], label=f'Val {metric}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'metrics_plot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def load_model_and_metadata(model_dir):
    """
    Load model and its metadata
    
    Args:
        model_dir: Directory containing model and metadata
        
    Returns:
        model, class_indices, config
    """
    # Load model
    model_path = os.path.join(model_dir, 'best_model.keras')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'final_model.keras')
    
    model = tf.keras.models.load_model(model_path)
    
    # Load class indices
    class_indices_path = os.path.join(model_dir, 'class_indices.json')
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Load config
    config_path = os.path.join(model_dir, 'training_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return model, class_indices, config


def create_data_summary(data_dir, save_path=None):
    """
    Create a summary of the dataset
    
    Args:
        data_dir: Path to dataset directory
        save_path: Path to save summary
        
    Returns:
        Summary dictionary
    """
    class_counts = {}
    total_images = 0
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            image_files = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(image_files)
            class_counts[class_name] = count
            total_images += count
    
    summary = {
        'total_classes': len(class_counts),
        'total_images': total_images,
        'class_distribution': class_counts,
        'average_images_per_class': total_images / len(class_counts) if class_counts else 0
    }
    
    # Print summary
    print("\nDataset Summary:")
    print("=" * 60)
    print(f"Total Classes: {summary['total_classes']}")
    print(f"Total Images: {summary['total_images']}")
    print(f"Average Images per Class: {summary['average_images_per_class']:.1f}")
    print("\nClass Distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} images")
    
    # Save summary
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nSummary saved to: {save_path}")
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.bar(range(len(classes)), counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Dataset Class Distribution')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plot_path = save_path.replace('.json', '_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {plot_path}")
    else:
        plt.show()
    
    plt.close()
    
    return summary

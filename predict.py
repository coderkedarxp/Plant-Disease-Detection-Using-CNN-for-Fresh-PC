"""
Prediction module for Plant Disease Classification
"""
import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from data_loader import PlantDiseaseDataLoader
import config


class PlantDiseasePredictor:
    """
    Plant disease predictor for inference on new images
    """
    
    def __init__(self, model_path, class_indices_path=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model (.keras or .h5)
            class_indices_path: Path to class indices JSON file
        """
        self.model_path = model_path
        self.model = None
        self.class_names = None
        self.class_indices = None
        
        # Load model
        self.load_model()
        
        # Load class indices
        if class_indices_path is None:
            # Try to find class_indices.json in the same directory as model
            model_dir = os.path.dirname(model_path)
            class_indices_path = os.path.join(model_dir, 'class_indices.json')
        
        self.load_class_indices(class_indices_path)
        
        # Initialize data loader for preprocessing
        self.data_loader = PlantDiseaseDataLoader(
            data_dir='',  # Not needed for prediction
            image_size=config.IMAGE_SIZE
        )
        
    def load_model(self):
        """Load trained model"""
        print(f"Loading model from: {self.model_path}")
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")
    
    def load_class_indices(self, class_indices_path):
        """
        Load class indices mapping
        
        Args:
            class_indices_path: Path to class indices JSON file
        """
        if os.path.exists(class_indices_path):
            print(f"Loading class indices from: {class_indices_path}")
            with open(class_indices_path, 'r') as f:
                self.class_indices = json.load(f)
            
            # Reverse mapping for predictions (index -> class name)
            self.class_names = {v: k for k, v in self.class_indices.items()}
            print(f"Loaded {len(self.class_names)} classes")
        else:
            print(f"Warning: Class indices file not found: {class_indices_path}")
            print("Using default class names from config")
            self.class_names = {i: name for i, name in enumerate(config.CLASS_NAMES)}
    
    def predict_image(self, image_path, top_k=3):
        """
        Predict disease for a single image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Load and preprocess image
        img = self.data_loader.load_and_preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img, verbose=0)
        
        # Get top k predictions
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        results = {
            'predictions': [],
            'image_path': image_path
        }
        
        for idx in top_indices:
            class_name = self.class_names.get(idx, f"Class_{idx}")
            confidence = float(predictions[0][idx])
            
            results['predictions'].append({
                'class': class_name,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.2f}%"
            })
        
        return results
    
    def predict_batch(self, image_paths, top_k=3):
        """
        Predict disease for multiple images
        
        Args:
            image_paths: List of image file paths
            top_k: Number of top predictions to return per image
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path, top_k)
                results.append(result)
            except Exception as e:
                print(f"Error predicting {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_from_array(self, image_array, top_k=3):
        """
        Predict disease from numpy array (useful for web apps)
        
        Args:
            image_array: RGB image as numpy array
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Resize and normalize
        img = cv2.resize(image_array, config.IMAGE_SIZE)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img, verbose=0)
        
        # Get top k predictions
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        results = {
            'predictions': []
        }
        
        for idx in top_indices:
            class_name = self.class_names.get(idx, f"Class_{idx}")
            confidence = float(predictions[0][idx])
            
            results['predictions'].append({
                'class': class_name,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.2f}%"
            })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with image and results
        
        Args:
            image_path: Path to image file
            save_path: Path to save visualization (optional)
        """
        import matplotlib.pyplot as plt
        
        # Get prediction
        result = self.predict_image(image_path, top_k=5)
        
        # Load original image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Display image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Input Image')
        
        # Display predictions
        predictions = result['predictions']
        classes = [p['class'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        y_pos = np.arange(len(classes))
        ax2.barh(y_pos, confidences, align='center', color='green', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(classes)
        ax2.invert_yaxis()
        ax2.set_xlabel('Confidence')
        ax2.set_title('Top Predictions')
        ax2.set_xlim([0, 1])
        
        # Add confidence percentages
        for i, v in enumerate(confidences):
            ax2.text(v + 0.02, i, f'{v*100:.1f}%', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Predict Plant Disease from Image')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.keras or .h5)')
    parser.add_argument('--image_path', type=str,
                       help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str,
                       help='Path to directory of images for batch prediction')
    parser.add_argument('--class_indices', type=str,
                       help='Path to class indices JSON file')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Directory to save visualization results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.image_dir:
        print("Error: Please provide either --image_path or --image_dir")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found: {args.model_path}")
        return
    
    # Initialize predictor
    print("\n" + "=" * 60)
    print("PLANT DISEASE PREDICTION")
    print("=" * 60)
    
    predictor = PlantDiseasePredictor(
        model_path=args.model_path,
        class_indices_path=args.class_indices
    )
    
    # Single image prediction
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: Image not found: {args.image_path}")
            return
        
        print(f"\nPredicting disease for: {args.image_path}")
        
        if args.visualize:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, 
                                      f"prediction_{Path(args.image_path).stem}.png")
            result = predictor.visualize_prediction(args.image_path, output_path)
        else:
            result = predictor.predict_image(args.image_path, top_k=args.top_k)
        
        print("\nPrediction Results:")
        print("-" * 60)
        for i, pred in enumerate(result['predictions'], 1):
            print(f"{i}. {pred['class']}: {pred['confidence_percent']}")
        
    # Batch prediction
    elif args.image_dir:
        if not os.path.exists(args.image_dir):
            print(f"Error: Directory not found: {args.image_dir}")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(Path(args.image_dir).glob(f"*{ext}"))
            image_paths.extend(Path(args.image_dir).glob(f"*{ext.upper()}"))
        
        if not image_paths:
            print(f"No images found in: {args.image_dir}")
            return
        
        print(f"\nPredicting diseases for {len(image_paths)} images...")
        
        results = predictor.predict_batch([str(p) for p in image_paths], 
                                         top_k=args.top_k)
        
        # Display results
        print("\nBatch Prediction Results:")
        print("=" * 60)
        for result in results:
            if 'error' in result:
                print(f"\n{result['image_path']}: ERROR - {result['error']}")
            else:
                print(f"\n{Path(result['image_path']).name}:")
                for i, pred in enumerate(result['predictions'], 1):
                    print(f"  {i}. {pred['class']}: {pred['confidence_percent']}")
        
        # Save results to JSON
        output_json = os.path.join(args.output_dir, 'batch_predictions.json')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {output_json}")
    
    print("\n" + "=" * 60)
    print("Prediction completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

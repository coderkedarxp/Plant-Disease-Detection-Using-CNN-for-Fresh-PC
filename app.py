"""
Streamlit Web App for Plant Disease Prediction
Run with: streamlit run app.py
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from predict import PlantDiseasePredictor
import config


# Page configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2e7d32;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558b2f;
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f1f8e9;
        margin: 1rem 0;
        color: #1b5e20;
    }
    .prediction-box h2, .prediction-box h3 {
        color: #1b5e20;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence-medium {
        color: #f57c00;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence-low {
        color: #d32f2f;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor(model_path, class_indices_path=None):
    """Load predictor model (cached)"""
    try:
        predictor = PlantDiseasePredictor(model_path, class_indices_path)
        return predictor, None
    except Exception as e:
        return None, str(e)


def get_confidence_color(confidence):
    """Get color class based on confidence level"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"


def display_prediction_results(results):
    """Display prediction results in a formatted way"""
    st.markdown("### 🔍 Prediction Results")
    
    if not results['predictions']:
        st.warning("No predictions available")
        return
    
    # Top prediction
    top_pred = results['predictions'][0]
    confidence_class = get_confidence_color(top_pred['confidence'])
    
    st.markdown(f"""
    <div class='prediction-box'>
        <h3>Top Prediction</h3>
        <h2>{top_pred['class']}</h2>
        <p class='{confidence_class}'>Confidence: {top_pred['confidence_percent']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # All predictions
    if len(results['predictions']) > 1:
        st.markdown("#### Other Possible Diseases")
        
        for i, pred in enumerate(results['predictions'][1:], 2):
            confidence_class = get_confidence_color(pred['confidence'])
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i}. {pred['class']}**")
            with col2:
                st.markdown(f"<p class='{confidence_class}'>{pred['confidence_percent']}</p>", 
                          unsafe_allow_html=True)


def main():
    # Header
    st.markdown("<h1 class='main-header'>🌿 Plant Disease Detector</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Upload an image of a plant leaf to detect diseases</p>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("Model Configuration")
        
        # Find available models
        models_dir = config.SAVED_MODELS_DIR
        available_models = []
        
        if os.path.exists(models_dir):
            for model_dir in os.listdir(models_dir):
                model_path = os.path.join(models_dir, model_dir, 'best_model.keras')
                if os.path.exists(model_path):
                    available_models.append((model_dir, model_path))
        
        if available_models:
            model_names = [name for name, _ in available_models]
            selected_model_name = st.selectbox(
                "Select Model",
                options=model_names,
                index=0
            )
            model_path = [path for name, path in available_models if name == selected_model_name][0]
            class_indices_path = os.path.join(os.path.dirname(model_path), 'class_indices.json')
        else:
            st.warning("No trained models found!")
            st.info("Please train a model first using train.py")
            model_path = st.text_input("Or enter model path manually:")
            class_indices_path = st.text_input("Class indices path (optional):")
        
        # Prediction settings
        st.subheader("Prediction Settings")
        top_k = st.slider("Number of predictions", min_value=1, max_value=10, value=5)
        
        st.markdown("---")
        
        # Information
        st.subheader("ℹ️ About")
        st.info("""
        This app uses deep learning to detect plant diseases from leaf images.
        
        **Supported formats:**
        - JPG, JPEG, PNG
        
        **Tips for best results:**
        - Use clear, well-lit images
        - Focus on the diseased area
        - Avoid blurry images
        """)
    
    # Main content
    if not model_path or not os.path.exists(model_path):
        st.error("⚠️ No model available. Please train a model first or specify a valid model path.")
        st.code("""
        # To train a model, run:
        python train.py --data_dir data/raw --epochs 50
        """)
        return
    
    # Load predictor
    predictor, error = load_predictor(model_path, class_indices_path)
    
    if error:
        st.error(f"Error loading model: {error}")
        return
    
    st.success(f"✅ Model loaded successfully from: {selected_model_name if available_models else model_path}")
    
    # File upload
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📤 Upload Plant Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf"
    )
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 Uploaded Image")
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Image info
            st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            st.markdown("#### 🧠 Analysis")
            
            # Convert PIL to numpy array
            image_array = np.array(image)
            
            # Convert RGBA to RGB if necessary
            if image_array.shape[-1] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                try:
                    results = predictor.predict_from_array(image_array, top_k=top_k)
                    display_prediction_results(results)
                    
                    # Download results button
                    results_json = json.dumps(results, indent=4)
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=results_json,
                        file_name="prediction_results.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    else:
        # Show sample instructions
        st.info("👆 Please upload an image to get started")
        
        # Show example if available
        st.markdown("---")
        st.markdown("### 📸 Example")
        st.write("The model can detect various plant diseases from leaf images.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌿 Plant Disease Detector</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

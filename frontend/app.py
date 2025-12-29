import streamlit as st
import pandas as pd
import requests

# Page Config
st.set_page_config(
    page_title="Text Classification App",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner="Loading class metadata...")
def load_classes():
    response = requests.get("http://localhost:8000/classes")
    response.raise_for_status()
    return response.json()

def main():
    # Title
    st.title("Text Classification System")
    st.markdown("Classify customer support tickets into appropriate queues")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This app uses a fine-tuned BERT model to classify "
            "customer support text into different categories."
        )

        st.header("⚙️ Settings")
        top_k = st.slider("Number of predictions to show", 1,3,5)
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)

    # Main content
    tab1, tab2 = st.tabs(["🔍 Prediction", "📈 Statistics"])

    # ==================== TAB 1: Prediction ====================
    with tab1:
        st.header("Text Classification")

        # Text Input
        text_input = st.text_area(
            "Enter text to classify:",
            height=150,
            placeholder="Example: I have a problem with my billing statement..."
        )

        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            predict_button  = st.button("Predict", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()

        if predict_button and text_input:
            with st.spinner("Analyzing..."):
                payload = {
                    "text": text_input,
                    "top_k": top_k
                }
                # Make Prediction
                response = requests.post(
                    "http://localhost:8000/predict",
                    json=payload
                )

                result = response.json()

                if 'error' in result and result['error']:
                    st.error(result['error'])
                else:
                    # Display results
                    st.success("✓ Classification complete!")                    
                    
                    st.markdown("### 🎯 Predicted Class")
                    confidence = result['confidence']
                    emoji = "🟢" if confidence > confidence_threshold else "🟡"
                    st.markdown(
                        f"{emoji} {result['predicted_class']}"
                    )
                    st.markdown(f"Confidence: {confidence:.2%}")

                # All predictions
                st.markdown("### 📋 All Predictions")
                predictions_df = pd.DataFrame(result['all_predictions'])
                predictions_df['probability'] = predictions_df['probability'].apply(lambda x: f"{x:.2%}")

                # Hide the index
                st.dataframe(predictions_df, hide_index=True)

        elif predict_button:
            st.warning("⚠️ Please enter some text to classify")
    
    with tab2:
        st.header("Model Statistics")

        col1, col2 = st.columns(2)

        class_info = load_classes()
        class_names = class_info["classes"]
        num_classes = class_info["num_classes"]

        with col1:
            st.metric("Number of classes", len(class_names))
        with col2:
            st.metric("Model Type", "BERT-base")
        
        st.markdown("### 📋 Available Classes")

        classes_df = pd.DataFrame({
            'Class ID': range(num_classes),
            'Class Name': class_names
        })

        st.dataframe(classes_df, use_container_width=True, hide_index=True)
        
if __name__ == "__main__":
    main()
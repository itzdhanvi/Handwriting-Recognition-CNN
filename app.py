import os
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

from preprocessing import preprocess_user_image

MODEL_PATH = "models/handwriting_cnn.keras"

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Handwriting Recognition",
    page_icon="✍️",
    layout="wide",
)

# --------------------------------------------------
# LOAD TRAINED CNN MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. First run: python train_model.py"
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# --------------------------------------------------
# CLASS LABELS
# --------------------------------------------------

# Must match the exact class-index order used during training.
CLASS_NAMES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t",
]

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
        color: #64748b;
    }

    .prediction {
        text-align: center;
        font-size: 80px;
        font-weight: 700;
        color: #0d9488;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">✍️ Handwritten Character Recognition</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">CNN-based recognition</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Model Information")
st.sidebar.write("**Model:** Convolutional Neural Network")
st.sidebar.write("**Input:** 28 × 28 grayscale")
st.sidebar.write("**Classes:** 47")

st.sidebar.markdown("---")
st.sidebar.info(
    "Draw a single handwritten letter or digit on the canvas, "
    "then click Recognize Character."
)

# --------------------------------------------------
# CHARACTER PREDICTION
# --------------------------------------------------

def predict(image):
    processed = preprocess_user_image(image)

    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            "The model output count does not match the 47 class labels."
        )

    index = int(np.argmax(probabilities))

    return (
        CLASS_NAMES[index],
        float(probabilities[index]),
        probabilities,
    )

# --------------------------------------------------
# DRAWING CANVAS
# --------------------------------------------------

st.subheader("Draw a Character")
st.caption(
    "Use your mouse or touchscreen to draw one letter or digit. "
    "Write clearly in the center of the canvas."
)

col1, col2 = st.columns([1, 1])

with col1:
    canvas = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        return_image_data=True,
    )

    recognize_button = st.button(
        "🔍 Recognize Character",
        use_container_width=True,
        type="primary",
    )

    clear_button = st.button(
        "🗑️ Clear Canvas",
        use_container_width=True,
    )

    if clear_button:
        st.rerun()

# --------------------------------------------------
# RECOGNITION RESULTS
# --------------------------------------------------

with col2:
    st.subheader("Recognition Result")

    if recognize_button:
        if canvas.image_data is None or np.max(canvas.image_data) == 0:
            st.warning("Please draw a character before recognizing.")

        else:
            try:
                with st.spinner("Recognizing your handwriting..."):
                    character, confidence, probabilities = predict(
                        canvas.image_data
                    )

                st.markdown("### Predicted Character")

                st.markdown(
                    f'<div class="prediction">{character}</div>',
                    unsafe_allow_html=True,
                )

                st.success(
                    f"Confidence: {confidence * 100:.2f}%"
                )

                # ------------------------------------------
                # TOP 5 PREDICTIONS
                # ------------------------------------------

                st.markdown("### Top 5 Predictions")

                top_indices = np.argsort(probabilities)[-5:][::-1]

                for idx in top_indices:
                    label = CLASS_NAMES[idx]
                    score = float(probabilities[idx])

                    st.write(
                        f"**{label}** — {score * 100:.2f}%"
                    )

                    st.progress(
                        min(max(score, 0.0), 1.0)
                    )

            except Exception:
                st.error(
                    "Recognition failed. Check preprocessing.py "
                    "and the model input format."
                )
                st.exception(Exception("See the error details above."))

    else:
        st.info(
            "Your predicted character and confidence score "
            "will appear here."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")
st.caption(
    "Handwriting Recognition System | Powered by a Convolutional Neural Network"
)



import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Spam SMS Classifier",
    page_icon="📩"
)

@st.cache_resource
def load_model():
    model_path = Path("outputs/spam_classifier.joblib")
    return joblib.load(model_path)

model = load_model()

st.title("📩 Spam SMS Classifier")
st.write(
    "Enter an SMS message to check whether it is Spam or Ham."
)

message = st.text_area(
    "Enter your SMS message:",
    placeholder="Type or paste an SMS here...",
    height=150
)

if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]

        classes = list(model.classes_)
        spam_index = classes.index(1) if 1 in classes else classes.index("spam")
        spam_probability = float(probabilities[spam_index])

        is_spam = prediction == 1 or prediction == "spam"

        label = "Spam" if is_spam else "Ham"
        confidence = spam_probability if is_spam else 1 - spam_probability

        if is_spam:
            st.error("Prediction: Spam")
        else:
            st.success("Prediction: Ham (Not Spam)")

        st.write(f"Confidence: {confidence * 100:.2f}%")
        st.caption(
            "This prediction is based on a machine learning model "
            "and may not always be correct."
        )

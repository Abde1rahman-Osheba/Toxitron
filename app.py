
import streamlit as st
from PIL import Image

from imagecaption import generate_caption
from text_classifier import TextClassifier
from database import save_record, load_records


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Text & Image Classifier",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cache the classifier so it loads only once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading classification models … this may take a few minutes on first run.")
def get_classifier():
    return TextClassifier()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Choose the what you want to check")
page = st.sidebar.radio(
    "Go to",
    ["Text", "Image", "View Database"],
)


# ===========================================================================
# PAGE 1 — Classify Text
# ===========================================================================
if page == "Classify Text":
    st.title("Text Classification")
    st.markdown(
        "Enter any text below and the hybrid ensemble model will classify it."
    )

    user_text = st.text_area(
        "Enter your text:",
        height=150,
        placeholder="Type or paste your text here …",
    )

    if st.button("Classify Text", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            classifier = get_classifier()
            with st.spinner("Classifying …"):
                result = classifier.classify(query=user_text)

            # Display results
            st.success(f"**Classification:** {result['label']}")
            st.metric("Confidence", f"{result['confidence']:.2%}")

            st.subheader("Class Probabilities")
            prob_data = result["probs"]
            st.bar_chart(prob_data)

            # Save to CSV database
            save_record(
                input_type="text",
                input_text=user_text.strip(),
                classification_label=result["label"],
                confidence=result["confidence"],
            )
            st.info("Result saved to database.")


# ===========================================================================
# PAGE 2 — Classify Image
# ===========================================================================
elif page == "Classify Image":
    st.title("Image Classification")
    st.markdown(
        "Upload an image. A caption will be generated using **BLIP-1**, "
        "then the caption is classified by the hybrid ensemble."
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Caption & Classify", type="primary"):
            # Generate caption
            with st.spinner("Generating caption with BLIP-1 …"):
                caption = generate_caption(image)

            st.subheader("Generated Caption")
            st.write(f"**{caption}**")

            # Classify caption
            classifier = get_classifier()
            with st.spinner("Classifying caption …"):
                result = classifier.classify(query="", image_caption=caption)

            # Display results
            st.success(f"**Classification:** {result['label']}")
            st.metric("Confidence", f"{result['confidence']:.2%}")

            st.subheader("Class Probabilities")
            prob_data = result["probs"]
            st.bar_chart(prob_data)

            # Save to CSV database
            save_record(
                input_type="image",
                input_text=caption,
                classification_label=result["label"],
                confidence=result["confidence"],
            )
            st.info("Result saved to database.")


# ===========================================================================
# PAGE 3 — View Database
# ===========================================================================
elif page == "View Database":
    st.title("Classification Database")
    st.markdown("All user inputs and their classification results are stored here.")

    records = load_records()

    if records.empty:
        st.info("No records yet. Classify some text or images to populate the database.")
    else:
        st.metric("Total Records", len(records))
        st.dataframe(records, use_container_width=True)

        # Download button
        csv_data = records.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="classifications_db.csv",
            mime="text/csv",
        )

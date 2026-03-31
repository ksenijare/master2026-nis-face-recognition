import streamlit as st
import numpy as np

from face_pipeline import (
    pil_to_rgb_array,
    analyze_face,
    cosine_similarity,
    similarity_to_score,
    decide,
    variance_of_laplacian_blur_score,
    draw_bbox,
)

st.set_page_config(page_title="Photo Similarity Checker", layout="wide")
st.title("AI Assistant for Document Photo Comparison")

st.write("""
This system automatically compares **two photos** of the same person (e.g., an old and a new photo for a document)  
and evaluates whether the old photo is suitable for use.  
The goal is to assist in deciding if a new photo is required for official documents.
""")

# Section: Upload Photos
st.header("Upload Photos")
st.write("Please upload the old and new photos to compare their similarity.")
col1, col2 = st.columns(2)
with col1:
    old_file = st.file_uploader("Old Photo", type=["jpg", "jpeg", "png"], key="old")
with col2:
    new_file = st.file_uploader("New Photo", type=["jpg", "jpeg", "png"], key="new")

# Section: Model Settings
st.header("Model Settings")
st.write("Adjust the options below to configure the face recognition model and similarity threshold.")
model_name = st.selectbox(
    "Face Embedding Model",
    ["ArcFace", "Facenet512", "Facenet", "VGG-Face"],
    index=0,
    help="Select a pre-trained model for generating face embeddings."
)
detector_backend = st.selectbox(
    "Face Detection Engine",
    ["retinaface", "mtcnn", "opencv", "ssd", "dlib"],
    index=0,
    help="Recommended: 'retinaface' for the best accuracy. OpenCV works only for very clear faces."
)
threshold = st.slider(
    "Similarity Threshold (Cosine Similarity)",
    0.20, 0.90, 0.45, 0.01,
    help="Set the similarity threshold: above the threshold - the old photo is acceptable; below - a new photo is required."
)
with st.expander("What is the similarity threshold?"):
    st.write(
        """
        The threshold determines the minimum similarity required for the system to recommend the old photo for use.
        - If similarity is **greater than or equal to** the threshold → Recommendation: **OK**
        - If similarity is lower → Recommendation: **New photo required**
        Adjust the threshold based on the model, image quality, and system requirements.
        """
    )

# Section: Analyze Photos
st.header("Analyze Photos")
st.write("Click the button below to analyze the uploaded photos.")
if st.button("Run Analysis", type="primary"):
    if not old_file or not new_file:
        st.error("Please upload both the old and new photos.")
        st.stop()

    old_img = pil_to_rgb_array(old_file)
    new_img = pil_to_rgb_array(new_file)

    st.subheader("Uploaded Photos")
    c1, c2 = st.columns(2)
    c1.image(old_img, caption="Old Photo", use_container_width=True)
    c2.image(new_img, caption="New Photo", use_container_width=True)

    st.subheader("Image Quality Check")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown(f"- **Resolution (Old):** {old_img.shape[1]}×{old_img.shape[0]}")
        st.markdown(f"- **Sharpness (Blur Score):** {variance_of_laplacian_blur_score(old_img):.1f} *(higher is better)*")
    with q2:
        st.markdown(f"- **Resolution (New):** {new_img.shape[1]}×{new_img.shape[0]}")
        st.markdown(f"- **Sharpness (Blur Score):** {variance_of_laplacian_blur_score(new_img):.1f} *(higher is better)*")

    try:
        old_res = analyze_face(
            old_img,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True
        )
        new_res = analyze_face(
            new_img,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True
        )

        with st.expander("Embedding Data (First 5 Values)"):
            st.write("Old Photo:", old_res.embedding[:5])
            st.write("New Photo:", new_res.embedding[:5])
            st.write("Are embeddings identical?", np.allclose(old_res.embedding, new_res.embedding))

        st.subheader("Face Crops")
        fc1, fc2 = st.columns(2)
        fc1.image(old_res.face_crop_rgb, caption="Old Photo Face Crop", use_container_width=True)
        fc2.image(new_res.face_crop_rgb, caption="New Photo Face Crop", use_container_width=True)

        st.subheader("Bounding Boxes on Original Photos")
        bb1, bb2 = st.columns(2)
        if old_res.bbox is not None:
            bb1.image(draw_bbox(old_img, old_res.bbox), caption="Old Photo (Bounding Box)", use_container_width=True)
        else:
            bb1.info("Bounding box not available.")
            bb1.image(old_img, caption="Old Photo (No Box)", use_container_width=True)

        if new_res.bbox is not None:
            bb2.image(draw_bbox(new_img, new_res.bbox), caption="New Photo (Bounding Box)", use_container_width=True)
        else:
            bb2.info("Bounding box not available.")
            bb2.image(new_img, caption="New Photo (No Box)", use_container_width=True)

        sim = cosine_similarity(old_res.embedding, new_res.embedding)
        score_0_100 = similarity_to_score(sim)
        recommendation = decide(sim, threshold)

        st.subheader("Comparison Results")
        r1, r2, r3 = st.columns(3)
        r1.metric("Cosine Similarity", f"{sim:.4f}")
        r2.metric("Score (0–100)", f"{score_0_100:.1f}")
        r3.metric("Recommendation", "OK" if recommendation == "OK" else "New Photo Required")
        
        if recommendation == "OK":
            st.success("The old photo is acceptable for use in documents.")
        else:
            st.warning("A new photo is recommended due to low similarity.")

        with st.expander("Technical Details"):
            st.markdown(f"- **Model:** {model_name}")
            st.markdown(f"- **Detector:** {detector_backend}")
            st.markdown(f"- **Embedding Dimension:** {old_res.embedding.shape[0]}")
            st.markdown(f"- **Decision Threshold:** {threshold:.2f}")
            st.markdown(f"- **Note:** This is a demo system and should not be used for automated decisions without human oversight.")

    except Exception as e:
        st.error("Analysis failed. Possible reasons: no face detected in the image or poor image quality. Try using a clearer photo.")
        st.code(str(e))

st.markdown("""
---
🔒 **Privacy Notice:** This system uses photos exclusively for demo/testing purposes. Photos are not stored or shared with third parties.  
⚠️ **Limitations:** This is a demonstration system and does not guarantee accuracy. Always visually verify results before making any decisions.
""")
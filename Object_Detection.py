import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_NAME = "yolo11n.pt"
OUTPUT_DIR = Path("output_images")
OUTPUT_DIR.mkdir(exist_ok=True)

@st.cache_resource
def load_detection_model():
    return YOLO(MODEL_NAME)

@st.cache_resource
def load_classification_model():
    # YOLO11 classification model used only to provide Top-3 class suggestions.
    return YOLO("yolo11n-cls.pt")


def top3_classification(crop):
    """Return up to three classification labels with probabilities for a crop."""
    try:
        model = load_classification_model()
        result = model.predict(source=np.array(crop.convert("RGB")), verbose=False)[0]
        probs = result.probs
        if probs is None:
            return "N/A"

        top_indices = probs.top5[:3]
        return "; ".join(
            f"{result.names[int(i)]} ({float(probs.data[int(i)]) * 100:.1f}%)"
            for i in top_indices
        )
    except Exception:
        return "N/A"


def detect_objects(image, confidence):
    model = load_detection_model()
    rgb = np.array(image.convert("RGB"))

    # Higher image size and test-time augmentation can help with small/unclear objects.
    results = model.predict(
        source=rgb,
        conf=confidence,
        imgsz=960,
        augment=True,
        verbose=False,
    )
    result = results[0]
    annotated = Image.fromarray(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))

    rows = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            crop = image.crop((max(0, x1i), max(0, y1i), min(image.width, x2i), min(image.height, y2i)))

            rows.append({
                "Object": result.names[int(cls_id)],
                "Confidence": round(float(conf) * 100, 2),
                "Top 3 Classification": top3_classification(crop),
                "X1": round(float(x1), 1),
                "Y1": round(float(y1), 1),
                "X2": round(float(x2), 1),
                "Y2": round(float(y2), 1),
            })

    return annotated, pd.DataFrame(
        rows,
        columns=["Object", "Confidence", "Top 3 Classification", "X1", "Y1", "X2", "Y2"],
    )


st.set_page_config(page_title="Object Detection System", page_icon="🔎", layout="wide")
st.title("🔎 Object Detection in Image")
st.caption("BCA Minor Project — Object Detection using Python, YOLO11 and Streamlit")

with st.sidebar:
    st.header("Detection Settings")
    confidence = st.slider("Confidence Threshold", 0.10, 0.95, 0.25, 0.05)
    st.info("Detections below the selected confidence threshold are filtered out.")
 #   st.markdown("**Improvement:** YOLO11 detection uses a larger image size and test-time augmentation. Top-3 classification is shown for each detected crop.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.markdown("""
    ### How to use
    1. Upload an image.
    2. Select a confidence threshold.
    3. Click **Detect Objects**.
    4. View bounding boxes, object names, confidence scores and Top-3 classification suggestions.
    """)
else:
    image = Image.open(uploaded).convert("RGB")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with c2:
        st.subheader("Detected Image")

    if st.button("🚀 Detect Objects", type="primary"):
        with st.spinner("Running YOLO11 object detection and Top-3 classification..."):
            annotated, df = detect_objects(image, confidence)

        with c2:
            st.image(annotated, use_container_width=True)

        st.subheader("Detection Results")

        if df.empty:
            st.warning("No objects were detected above the selected threshold.")
        else:
            a, b, c = st.columns(3)
            a.metric("Objects Detected", len(df))
            b.metric("Unique Classes", df["Object"].nunique())
            c.metric("Average Confidence", f"{df['Confidence'].mean():.2f}%")

            st.dataframe(df, use_container_width=True)

            buf = io.BytesIO()
            annotated.save(buf, format="JPEG", quality=95)
            st.download_button(
                "⬇️ Download Detected Image",
                buf.getvalue(),
                "detected_image.jpg",
                "image/jpeg",
            )

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Detection Results (CSV)",
                csv_data,
                "detection_results.csv",
                "text/csv",
            )

        out = OUTPUT_DIR / "latest_detection.jpg"
        annotated.save(out, quality=95)
        st.success(f"Result saved to: {out}")

st.divider()
st.markdown("**Workflow:** Input Image → Preprocessing → YOLO11 Detection → Bounding Boxes → Object Names & Confidence → Top-3 Classification → Visualization")
#st.caption("Evaluation note: model accuracy/mAP is not calculated in this version. Confidence scores and classification probabilities are prediction outputs, not accuracy metrics.")
#st.caption("Note: Top-3 classification is supplementary. It can suggest likely labels for a detected crop, but it cannot guarantee that a missed or incorrectly detected object will be corrected. For higher accuracy on specific project classes, custom training with a labelled dataset is recommended.")

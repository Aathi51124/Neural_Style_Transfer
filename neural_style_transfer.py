import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import matplotlib.pyplot as plt
import tempfile
import cv2

tf.executing_eagerly()

st.set_page_config(
    page_title="Neural Style Transfer", layout="wide"
)

def load_image(image_buffer, image_size=(1024, 1024)):
    img = plt.imread(image_buffer).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.0
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def export_image(tf_img):
    pil_image = Image.fromarray(np.squeeze(tf_img * 255).astype(np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG", quality=95)
    byte_image = buffer.getvalue()
    return byte_image

def apply_style_transfer_to_frame(frame, stylize_model, style_image):
    frame = frame.astype(np.float32) / 255.0
    frame = tf.convert_to_tensor(frame[np.newaxis, ...])
    frame = tf.image.resize(frame, (1024, 1024), preserve_aspect_ratio=True)
    results = stylize_model(tf.constant(frame), tf.constant(style_image))
    stylized_frame = results[0].numpy()
    return (stylized_frame[0] * 255).astype(np.uint8)

def process_video(video_path, stylize_model, style_image, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        styled_frame = apply_style_transfer_to_frame(frame, stylize_model, style_image)
        styled_frame = cv2.resize(styled_frame, (width, height))
        out.write(styled_frame)

    cap.release()
    out.release()

def st_ui():
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = []

    if "result_history" not in st.session_state:
        st.session_state.result_history = []

    image_upload1 = st.sidebar.file_uploader("Load your content image", type=["jpeg", "png", "jpg"], key="content_image", help="Upload the image you want to style")
    image_upload2 = st.sidebar.file_uploader("Load your style image", type=["jpeg", "png", "jpg"], key="style_image", help="Upload the style image")
    video_upload = st.sidebar.file_uploader("Load your content video", type=["mp4", "avi"], key="content_video", help="Upload the video you want to style")

    col1, col2, col3 = st.columns(3)
    
    st.sidebar.title("Style Transfer")
    st.sidebar.markdown("Your personal neural style transfer")

    with st.spinner("Loading content image..."):
        if image_upload1 is not None:
            col1.header("Content Image")
            col1.image(image_upload1, use_column_width=True)
            original_image = load_image(image_upload1)
            st.session_state.upload_history.append({"type": "content", "image": image_upload1.getvalue()})
        else:
            original_image = load_image("1.jpg")
    
    with st.spinner("Loading style image..."):
        if image_upload2 is not None:
            col2.header("Style Image")
            col2.image(image_upload2, use_column_width=True)
            style_image = load_image(image_upload2)
            st.session_state.upload_history.append({"type": "style", "image": image_upload2.getvalue()})
        else:
            style_image = load_image("2.jpg")
    
    if st.sidebar.button(label="Start Styling Image"):
        if image_upload1 and image_upload2:
            with st.spinner('Generating stylized image...'):
                stylize_model = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
                results = stylize_model(tf.constant(original_image), tf.constant(style_image))
                stylized_photo = results[0]
                col3.header("Final Image")
                col3.image(np.array(stylized_photo))
                st.session_state.result_history.append(export_image(stylized_photo))
                st.download_button(label="Download Final Image", data=export_image(stylized_photo), file_name="stylized_image.png", mime="image/png")
        else:
            st.sidebar.warning("Please upload both content and style images.")
    
    if st.sidebar.button(label="Start Styling Video"):
        if video_upload and image_upload2:
            with st.spinner('Generating stylized video...'):
                stylize_model = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
                temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                process_video(video_upload, stylize_model, style_image, temp_video_path)

                with open(temp_video_path, "rb") as f:
                    st.sidebar.download_button(label="Download Stylized Video", data=f, file_name="stylized_video.mp4", mime="video/mp4")

        else:
            st.sidebar.warning("Please upload both content video and style image.")
    
    st.sidebar.subheader("Upload History")
    for idx, upload in enumerate(st.session_state.upload_history):
        st.sidebar.markdown(f"*Upload {idx+1} ({upload['type']})*") 
        st.sidebar.image(BytesIO(upload['image']), width=100)

    st.sidebar.subheader("Generated Results")
    for idx, result in enumerate(st.session_state.result_history):
        st.sidebar.markdown(f"*Result {idx+1}*")
        st.sidebar.image(BytesIO(result), width=100)

if __name__ == "__main__":
    st_ui()

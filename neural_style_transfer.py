import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import cv2
import tempfile
import os

tf.executing_eagerly()

st.set_page_config(page_title="Neural Style Transfer", layout="wide")

import io

def load_image(image_buffer, image_size=(1024, 1024)):
    if isinstance(image_buffer, str):
        # This handles the default image case; you can replace '1.jpg' with any image you want.
        img = Image.open(image_buffer)
    else:
        # Handle the uploaded file
        img = Image.open(io.BytesIO(image_buffer.read()))

    img = np.array(img).astype(np.float32)[np.newaxis, ...]
    
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

def apply_style_transfer_to_video(content_video_path, style_image, model, output_path):
    cap = cv2.VideoCapture(content_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a temporary file for the output video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to the required format for style transfer
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = np.array(img).astype(np.float32)[np.newaxis, ...] / 255.0
        img = tf.image.resize(img, (frame_height, frame_width))

        # Apply style transfer
        result = model(tf.constant(img), tf.constant(style_image))
        stylized_frame = result[0].numpy()

        # Convert the stylized frame back to the original format
        stylized_frame = np.squeeze(stylized_frame * 255).astype(np.uint8)
        stylized_frame = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        out.write(stylized_frame)

    cap.release()
    out.release()

def st_ui():
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = []
    if "result_history" not in st.session_state:
        st.session_state.result_history = []

    st.sidebar.title("Style Transfer")
    st.sidebar.markdown("Your personal neural style transfer")

    image_upload1 = st.sidebar.file_uploader("Load your content image", type=["jpeg", "png", "jpg"], key="content_image", help="Upload the image you want to style")
    image_upload2 = st.sidebar.file_uploader("Load your style image", type=["jpeg", "png", "jpg"], key="style_image", help="Upload the style image")
    video_upload = st.sidebar.file_uploader("Load your content video", type=["mp4", "mov"], key="content_video", help="Upload the video you want to style")

    col1, col2, col3 = st.columns(3)

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

    if st.sidebar.button(label="Start Styling for Image"):
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

    if st.sidebar.button(label="Start Styling for Video"):
        if video_upload and image_upload2:
            with st.spinner('Processing video...'):
                temp_dir = tempfile.mkdtemp()
                content_video_path = os.path.join(temp_dir, video_upload.name)
                with open(content_video_path, "wb") as f:
                    f.write(video_upload.read())

                output_video_path = os.path.join(temp_dir, "stylized_video.mp4")
                stylize_model = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

                apply_style_transfer_to_video(content_video_path, style_image, stylize_model, output_video_path)

                with open(output_video_path, "rb") as f:
                    st.download_button(label="Download Final Video", data=f, file_name="stylized_video.mp4", mime="video/mp4")
        else:
            st.sidebar.warning("Please upload a video and a style image.")

    st.sidebar.subheader("Upload History")
    for idx, upload in enumerate(st.session_state.upload_history):
        st.sidebar.markdown(f"Upload {idx+1} ({upload['type']})")
        st.sidebar.image(BytesIO(upload['image']), width=100)

    st.sidebar.subheader("Generated Results")
    for idx, result in enumerate(st.session_state.result_history):
        st.sidebar.markdown(f"Result {idx+1}")
        st.sidebar.image(BytesIO(result), width=100)

if __name__ == "__main__":
    st_ui()

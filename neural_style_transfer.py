import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import cv2
import tempfile

st.set_page_config(page_title="Video Neural Style Transfer", layout="wide")

# Load image function
def load_image(image_buffer, image_size=(512, 512)):
    img = Image.open(image_buffer)
    img = img.convert("RGB")
    img = img.resize(image_size)
    img = np.array(img).astype(np.float32)[np.newaxis, ...] / 255.0
    return img

# Export video function
def export_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

# Style transfer function
def apply_style_transfer(content_frame, style_image, model):
    content_tensor = tf.convert_to_tensor(content_frame)
    content_tensor = tf.image.resize(content_tensor, (512, 512))
    content_tensor = content_tensor[tf.newaxis, ...]

    style_tensor = tf.convert_to_tensor(style_image)
    style_tensor = tf.image.resize(style_tensor, (512, 512))
    style_tensor = style_tensor[tf.newaxis, ...]

    outputs = model(tf.constant(content_tensor), tf.constant(style_tensor))
    stylized_frame = outputs[0]
    stylized_frame = tf.image.resize(stylized_frame, (content_frame.shape[0], content_frame.shape[1]))
    return np.array(stylized_frame[0] * 255, dtype=np.uint8)

# Streamlit UI
def st_ui():
    st.title("Video Neural Style Transfer")
    st.sidebar.title("Upload and Configure")
    
    content_video = st.sidebar.file_uploader("Upload Content Video", type=["mp4", "avi", "mov"])
    style_choice = st.sidebar.radio("Choose Style Source", ("Style Image", "Style Video"))
    style_image = None
    style_video = None

    if style_choice == "Style Image":
        style_image = st.sidebar.file_uploader("Upload Style Image", type=["jpeg", "png", "jpg"])
    elif style_choice == "Style Video":
        style_video = st.sidebar.file_uploader("Upload Style Video", type=["mp4", "avi", "mov"])

    st.sidebar.write("Neural Style Transfer Settings")
    fps = st.sidebar.slider("Output FPS", 1, 60, 30)

    if st.sidebar.button("Start Style Transfer"):
        if content_video:
            with st.spinner("Processing Video..."):
                # Temporary file handling
                temp_content_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_content_file.write(content_video.read())
                temp_content_file.close()
                
                # Load the content video using OpenCV
                cap_content = cv2.VideoCapture(temp_content_file.name)
                frame_count = int(cap_content.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(cap_content.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap_content.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Load the style transfer model
                model = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

                # Load style image or video frames
                if style_image:
                    style_img = load_image(style_image)
                elif style_video:
                    temp_style_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    temp_style_file.write(style_video.read())
                    temp_style_file.close()
                    cap_style = cv2.VideoCapture(temp_style_file.name)

                # Process the video frame by frame
                frames = []
                frame_idx = 0

                while cap_content.isOpened():
                    ret, content_frame = cap_content.read()
                    if not ret:
                        break

                    content_frame_rgb = cv2.cvtColor(content_frame, cv2.COLOR_BGR2RGB)

                    if style_image:
                        # Apply style transfer with the static style image
                        stylized_frame = apply_style_transfer(content_frame_rgb, style_img, model)
                    elif style_video:
                        # Match frame from style video
                        ret_style, style_frame = cap_style.read()
                        if not ret_style:
                            cap_style.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret_style, style_frame = cap_style.read()
                        style_frame_rgb = cv2.cvtColor(style_frame, cv2.COLOR_BGR2RGB)
                        style_img = style_frame_rgb.astype(np.float32) / 255.0
                        stylized_frame = apply_style_transfer(content_frame_rgb, style_img, model)

                    # Convert back to BGR for saving
                    stylized_frame_bgr = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)
                    frames.append(stylized_frame_bgr)

                    frame_idx += 1
                    st.progress(frame_idx / frame_count)

                cap_content.release()
                if style_video:
                    cap_style.release()

                # Export the processed frames to a video
                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                export_video(frames, output_video_path, fps)

                # Display the final video
                st.video(output_video_path)
                st.success("Style Transfer Completed!")

if __name__ == "__main__":
    st_ui()

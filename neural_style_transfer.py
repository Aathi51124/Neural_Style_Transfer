import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, ReLU, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

tf.executing_eagerly()

st.set_page_config(
    page_title="CycleGAN Style Transfer", layout="wide"
)

def load_image(image_buffer, image_size=(256, 256)):
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
    pil_image.save(buffer, format="PNG")
    byte_image = buffer.getvalue()
    return byte_image

def generator_model():
    inputs = Input(shape=(256, 256, 3))
    x = Conv2D(64, kernel_size=7, strides=1, padding="same")(inputs)
    x = ReLU()(x)
    
    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
    x = ReLU()(x)
    
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
    x = ReLU()(x)
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x)
    x = ReLU()(x)
    
    outputs = Conv2D(3, kernel_size=7, strides=1, padding="same", activation="tanh")(x)
    return Model(inputs, outputs)

def discriminator_model():
    inputs = Input(shape=(256, 256, 3))
    x = Conv2D(64, kernel_size=4, strides=2, padding="same")(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Conv2D(1, kernel_size=4, strides=1, padding="same")(x)
    return Model(inputs, outputs)

def st_ui():
 
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = []

    if "result_history" not in st.session_state:
        st.session_state.result_history = []

    image_upload = st.sidebar.file_uploader("Load your image", type=["jpeg", "png", "jpg"], key="uploaded_image", help="Upload the image you want to transform using CycleGAN")
    
    col1, col2 = st.columns(2)
    
    st.sidebar.title("CycleGAN Style Transfer")
    st.sidebar.markdown("Transform images using a basic CycleGAN implementation")

   
    with st.spinner("Loading input image..."):
        if image_upload is not None:
            col1.header("Input Image")
            col1.image(image_upload, use_column_width=True)
            input_image = load_image(image_upload)

            st.session_state.upload_history.append({"type": "input", "image": image_upload.getvalue()})
        else:
            st.warning("Please upload an image to continue.")
            return

  
    G_XtoY = generator_model() 
    G_YtoX = generator_model() 
    D_X = discriminator_model() 
    D_Y = discriminator_model()  

    st.sidebar.subheader("Model Architectures")
    st.sidebar.text("Generator X -> Y")
    G_XtoY.summary(print_fn=lambda x: st.sidebar.text(x))
    st.sidebar.text("Generator Y -> X")
    G_YtoX.summary(print_fn=lambda x: st.sidebar.text(x))
    st.sidebar.text("Discriminator X")
    D_X.summary(print_fn=lambda x: st.sidebar.text(x))
    st.sidebar.text("Discriminator Y")
    D_Y.summary(print_fn=lambda x: st.sidebar.text(x))

    st.sidebar.info("Training setup is complete. For a full implementation, train the model using custom training loops.")
    st.sidebar.warning("Real-time training is computationally expensive and may not be suitable for Streamlit.")

if __name__ == "__main__":
    st_ui()

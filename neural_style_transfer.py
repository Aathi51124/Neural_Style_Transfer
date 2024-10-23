import streamlit as st
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import dnnlib
import legacy  
import sys
sys.path.append('./stylegan3')  
import dnnlib
import legacy

st.set_page_config(page_title="Advanced StyleGAN3 Generator", layout="wide")

def load_pretrained_model(network_pkl):
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) 
    return G

def generate_image(G, z, truncation_psi=0.7):
    #DOUBT
    img = G(z, None, truncation_psi=truncation_psi, noise_mode='const')
    img = (img + 1) * (255/2)  
    img = img.permute(0, 2, 3, 1).cpu().numpy()  
    img = np.clip(img[0], 0, 255).astype(np.uint8)  
    return Image.fromarray(img)

def export_image(image):
   
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    byte_image = buffer.getvalue()
    return byte_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

network_pkl = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/sg3-r-ffhq-1024x1024.pkl"
G = load_pretrained_model(network_pkl)

def st_ui():
    st.title("StyleGAN3 High-Resolution Image Generator")
    st.sidebar.title("StyleGAN3 Controls")

    seed = st.sidebar.number_input("Seed for Random Latent", min_value=0, max_value=100000, value=42, step=1)
    truncation_psi = st.sidebar.slider("Truncation Psi", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

    rnd = np.random.RandomState(seed)
    z = torch.from_numpy(rnd.randn(1, G.z_dim)).to(device)

    if st.sidebar.button("Generate Image"):
        with st.spinner('Generating image...'):
            generated_image = generate_image(G, z, truncation_psi)
            st.image(generated_image, caption="Generated Image", use_column_width=True)
            st.download_button(label="Download Image", data=export_image(generated_image), file_name="generated_image.png", mime="image/png")

if __name__ == "__main__":
    st_ui()    

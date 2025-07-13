
import streamlit as st
import torchvision.transforms as tt
from PIL import Image
from SRGAN import *
import io
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "img_idx" not in st.session_state:
    st.session_state.img_idx = 0
if "generator" not in st.session_state:
    st.session_state.generator = Generator()
    PATH = "SRGAN\generator_20250708_105415"
    state_dict = torch.load(PATH, weights_only=True, map_location=torch.device(DEVICE))
    st.session_state.generator.load_state_dict(state_dict)
if "discriminator" not in st.session_state:
    st.session_state.discriminator = Discriminator()
if "image" not in st.session_state:
     st.session_state.image = False


def change_img_idx():
    st.session_state.img_idx = (st.session_state.img_idx + 1) % len(image_filenames)

def denorm(img_tensors):
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    return img_tensors * stats[1][0] + stats[0][0]

############################################################
st.markdown("### 1. Data generation and preparation")
#
image_filenames = [ "SRGAN/000016.png" , "SRGAN/000008.png", "SRGAN/000027.png" ]  
image_size = 256
transform_HR=tt.Compose([tt.Resize(image_size),
                                 tt.CenterCrop(image_size)])
transform_LR=tt.Compose([tt.Resize(image_size // 4),
                                 tt.CenterCrop(image_size // 4)])
image = Image.open(image_filenames[st.session_state.img_idx])
#
colHR, colLR = st.columns(2)
Image_HR = transform_HR(image)
Image_LR = transform_LR(image)
with colHR:
    st.image(Image_HR, caption="High resolution image", use_container_width=True)

with colLR:
    st.image(Image_LR, caption="Low resolution image", use_container_width=True)

change_img_btn =  st.button(label = "Change image", type="secondary",on_click=change_img_idx, use_container_width=True)

#############################################################
st.markdown("### 2. Model loading and inference")

options = ["Discriminator", "Generator"]
selection = st.pills("Choose model to display",options, default=options[0] )
model_container = st.container(border=True, height= 500)
if selection == options[0]:
    model_container.code(st.session_state.discriminator)
if selection == options[1]:
    model_container.code(st.session_state.generator)

##############################################################
st.markdown("### 3. Model operation")

uploaded_file = st.file_uploader(
    label= "Choose the files to enhance", accept_multiple_files=False, label_visibility='hidden',
    type = ["jpg", "jpeg", "png"]
)

left, right = st.columns(spec=[0.9, 0.1], vertical_alignment='center', gap="medium")
img = None
with left:
    enhance_btn = st.button(label = "Enhance image", type="secondary", use_container_width=True )
with right:
    if enhance_btn and uploaded_file is not None:
        with st.spinner(text="", show_time=False, width="content", ):
                bytes_data = uploaded_file.read()
                stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                transform=tt.Compose([tt.Resize(image_size // 4),
                                    tt.CenterCrop(image_size // 4),
                                    tt.ToTensor(),
                                    tt.Normalize(*stats)])
                            
                img = transform(Image.open(io.BytesIO(bytes_data)))
                img = img.view(1, 3, 64, 64)
                norm = nn.Hardtanh(0, 1).to(DEVICE)
                generated_image = norm(st.session_state.generator(img)).cpu().detach()
                st.session_state.image = True

if st.session_state.image:
    st.session_state.image = False
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))
    img = img.view(3, 64, 64)
    generated_image = generated_image.view(3, 256, 256)

    axs[0].imshow(make_grid(denorm(img)).permute(1, 2, 0))
    axs[1].imshow(make_grid(generated_image).permute(1, 2, 0))
    st.pyplot(fig)

        

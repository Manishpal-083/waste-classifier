import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

device = "cpu"

st.set_page_config(page_title="Waste Classifier", page_icon="♻️")

st.title("♻️ Waste Classifier")

st.markdown("AI Model trained by Manish • upload image or use camera")

option = st.radio("", ["Upload Image", "Camera"])

# model load
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features,2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

idx2class = {0:"O", 1:"R"}

img = None

# upload block
if option == "Upload Image":
    file = st.file_uploader("Upload a waste image", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=300)

# camera block
elif option == "Camera":
    cam = st.camera_input("Use camera")
    if cam:
        img = Image.open(cam).convert("RGB")
        st.image(img, width=300)


# predict
if img is not None:
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)
        _,p = torch.max(pred,1)

    label = idx2class[p.item()]
    if label=="O":
        st.success("Prediction: **Organic Waste** ✅")
    else:
        st.success("Prediction: **Recyclable Waste** ♻️")

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

device = "cpu"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features,2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

idx2class = {0:"O", 1:"R"}

st.set_page_config(page_title="Waste Classifier")
st.title("♻️ Waste Classifier App")

img_file = st.file_uploader("Upload Waste Image", type=["jpg","jpeg","png"])

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)
        _,p = torch.max(pred,1)

    label = idx2class[p.item()]
    
    if label == "O":
        st.success("Prediction: **Organic Waste** ✅")
    else:
        st.success("Prediction: **Recyclable Waste** ♻️")

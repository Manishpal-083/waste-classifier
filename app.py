import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

st.set_page_config(page_title="♻️ Waste Classifier", page_icon="♻️")

st.markdown("""
<style>
    .title { font-size:35px !important; font-weight:700; text-align:center }
    .sub {opacity:0.6; text-align:center}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>AI Waste Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Organic vs Recyclable</div><br>", unsafe_allow_html=True)

device = "cpu"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# MODEL LOAD
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features,2)
model.load_state_dict(torch.load("model.pth",map_location="cpu"))
model.eval()

idx2class = {0:"Organic", 1:"Recyclable"}

col1,col2 = st.columns(2)

with col1:
    upload = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

with col2:
    cam = st.camera_input("Open Camera")

img = None
if upload: img = Image.open(upload).convert("RGB")
if cam: img = Image.open(cam).convert("RGB")

if img:
    st.image(img, width=300)

    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)
        probs = F.softmax(pred, dim=1)
        conf,p = torch.max(probs,1)

    label = idx2class[p.item()]
    confidence = conf.item()*100

    st.markdown(f"### Result: **{label} Waste**")
    st.progress(confidence/100)
    st.write(f"Model Confidence: `{confidence:.2f}%`")

    if label=="Organic":
        st.success("Compostable items → fruits, vegetables, food leftovers")
    else:
        st.warning("Recyclable items → plastic bottles, metals, glass etc")

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.optim import Adam

print("THIS IS FINAL TRAIN FILE ✅")

device = "cpu"
print("device =",device)

train_dir = "light_data"
test_dir = "data/TEST"   # <-- TEST capital folder

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

print("classes =", train_data.classes)
print("train =", len(train_data), "test =", len(test_data))

train_loader = torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,batch_size=32)

model = models.mobilenet_v2(weights="IMAGENET1K_V1")

for p in model.features.parameters():
    p.requires_grad = False

model.classifier[1] = nn.Linear(model.classifier[1].in_features,2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

def get_acc(loader):
    model.eval()
    c,t=0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            p = model(x)
            _,pr = torch.max(p,1)
            c += (pr==y).sum().item()
            t += y.size(0)
    return c/t

print("✅ TRAIN LOOP START")

for epoch in range(3):
    print(f"epoch {epoch+1}/3 starting...")
    model.train()
    loss_sum=0
    for imgs,labels in train_loader:
        imgs,labels = imgs.to(device),labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out,labels)
        loss.backward()
        optimizer.step()
        loss_sum+=loss.item()

    print(f"epoch {epoch+1}/3 loss={loss_sum/len(train_loader):.4f} train_acc={get_acc(train_loader):.3f} test_acc={get_acc(test_loader):.3f}")

torch.save(model.state_dict(),"model.pth")
print("✅ MODEL SAVED")

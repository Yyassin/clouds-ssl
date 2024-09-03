import os
from torchvision import transforms
import timm
import torch
from PIL import Image
import csv

from models.mocov3 import MoCoV3
from models.remoco import JEPA


# Load labels
data = []
with open(os.path.abspath('./annotations.csv'), 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        entry = []
        for i, elem in enumerate(row):
            if i == 0:
                entry.append(elem)
            else:
                entry.append(float(elem))
        data.append(entry)

# Load images
all_data = []
for i, item in enumerate(data):
    image = Image.open(os.path.abspath(f"./annotated_images/{item[0]}"))
    tensor = transforms.ToTensor()(image)
    annotation = item[1:]
    all_data.append([tensor, annotation])

# Load models
model_root_dir = "./saved_models"
model_paths = os.listdir(os.path.abspath(model_root_dir))
model_paths = [f"{model_root_dir}/{x}" for x in model_paths]
model_paths = [x for x in model_paths if not os.path.isdir(x)]

# Get embeddings
for model_name in model_paths:
    model = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=True).to("cuda:0")
    model = model.eval()
    is_remoco = "remoco" in model_name
    if is_remoco:
        pre = JEPA(enc=model, M=4, device="cuda:0").to("cuda:0")
    else:
        pre = MoCoV3(encoder=model, is_ViT=True, dim=384, device="cuda:0").to("cuda:0")
    pre = pre.eval()

    print(f"Loading model: {model_name}")
    pre.load_state_dict(torch.load(os.path.abspath(model_name), map_location="cuda:0"))
    
    all_embeds = []
    all_labels = []
    with torch.no_grad():
        for row in all_data:
            image, annotation = row
            image = image.unsqueeze(dim=0).to("cuda:0")
        
            if is_remoco:
                embeds = pre.student_encoder.forward_features(image)
            else:
                embeds = pre.encoder_q.forward_features(image)
            embeds = embeds[:, 1:, :].mean(dim=1)  # Global average pool
            all_embeds.append(embeds.cpu())
            all_labels.append(torch.tensor(annotation).unsqueeze(dim=0))
    
    all_embeds = torch.cat(all_embeds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    save_dir = "./eval/embeddings/"
    if not os.path.exists(os.path.abspath(save_dir)):
        os.makedirs(os.path.abspath(save_dir))
    torch.save(obj={"embeds": all_embeds, "labels": all_labels}, f=f'{save_dir}/{model_name.split("/")[-1]}')

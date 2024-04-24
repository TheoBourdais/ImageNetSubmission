# Set up device
import torch
from torch.utils.data import DataLoader
import timm
import os
from utils import TestDataset, get_test_submission


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare the data loaders and model

model = timm.create_model("vit_base_patch16_224", pretrained=True)

data_config = timm.data.resolve_model_data_config(model)
preprocess = timm.data.create_transform(**data_config, is_training=False)
model.to(device)


test_dataset = TestDataset("./imagenet/test/", transform=preprocess)
loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

submission = get_test_submission(model, loader, device)

# write submission to csv but without overwriting any files, adding (1), (2), etc. to the filename
if not os.path.exists("submission.txt"):
    submission.to_csv("submission.txt", index=False, header=False, sep=" ")
else:
    i = 1
    while os.path.exists(f"submission{i}.txt"):
        i += 1
    submission.to_csv(f"submission{i}.txt", index=False, header=False, sep=" ")

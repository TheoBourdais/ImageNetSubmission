# Simple code for preparing a submission to the ImageNet evaluation server for the test set
 
This code is a simple example of how to prepare an ImageNet submission to the evaluation server. A notebook is provided that shows how to use the code to prepare a submission. The folder `src` contains the code for preparing the submission, as well as the file [idx_to_ILSVRC_ID.csv](./src/idx_to_ILSVRC_ID.csv) which maps the ImageNet class index (that are given automatically when using `torchvision.datasets.ImageFolder`) to the ILSVRC ID needed for the submission. It assumes you have already downloaded the ImageNet test dataset, which can be found [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
Once the submission is prepared, you can submit it to the [ImageNet evaluation server](https://image-net.org/challenges/LSVRC/eval_server.php) to get the results.

> Note: There are more details on how to obtain the `idx_to_ILSVRC_ID.csv` file in the [README](./src/README.md) of the `src` folder.


## Usage

You may open the notebook [example.ipynb](example.ipynb) and modify it to prepare a submission. The notebook is self-contained and should be easy to follow.
You can also directly use the code below

```python
import torch
from torch.utils.data import DataLoader
import timm
import os
from src.utils import TestDataset, get_test_submission


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare the data loaders and model

model = timm.create_model("vit_base_patch16_224", pretrained=True)

data_config = timm.data.resolve_model_data_config(model)
preprocess = timm.data.create_transform(**data_config, is_training=False)
model.to(device)


test_dataset = TestDataset("./imagenet/test/", transform=preprocess)
loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Prepare the submission

submission = get_test_submission(model, loader, device)
submission.to_csv("submission.txt", index=False, header=False, sep=" ")

```

## License

Public domain.

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # necessary for cropped images in Imagenet

from PIL import Image
import torch
import glob
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm


class TestDataset(Dataset):
    def __init__(self, path, transform=None):
        self.image_paths = glob.glob(path + "*.JPEG")
        self.transform = transform

    def __getitem__(self, index):
        path = self.image_paths[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return path.split("/")[-1], x

    def __len__(self):
        return len(self.image_paths)


idx_to_ilsvrc_id = pd.read_csv(
    "https://github.com/TheoBourdais/ImageNetSubmission/raw/main/src/idx_to_ILSVRC_ID.csv"
)
idx_to_ilsvrc_id = {
    idx: ilsvrc_id
    for idx, ilsvrc_id in zip(idx_to_ilsvrc_id["idx"], idx_to_ilsvrc_id["ILSVRC_ID"])
}


def get_test_submission(model, dataloader, device):

    model.eval()
    # prepare a df for submission
    submission = pd.DataFrame(columns=[f"choice {i}" for i in range(1, 6)])
    steps = 0
    with torch.no_grad():
        for paths, images in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            # get top 5 predictions
            _, predicted = torch.topk(outputs, 5)
            submission_batch = pd.DataFrame(
                data=predicted.cpu().numpy(), columns=submission.columns, index=paths
            )
            submission = pd.concat([submission, submission_batch])
            steps += 1
    # sort submission by index
    submission = submission.sort_index()
    for i in range(1, 6):
        submission[f"choice {i}"] = submission[f"choice {i}"].apply(
            lambda x: idx_to_ilsvrc_id[x]
        )
    return submission

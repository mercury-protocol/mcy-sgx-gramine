import torch
import torchvision
from torch.utils.data import Dataset

from simulate_vulkan.user_script import SplitMNISTDataSet


if __name__ == "__main__":
    from simulate_vulkan.data_manipulation.constants import SPLIT_DATA_PATH, BATCH_SIZE_TRAIN, NORMALIZE_MEAN, NORMALIZE_STD
    from simulate_vulkan.helpers.eval import check_data

    custom_dataset = SplitMNISTDataSet(
        SPLIT_DATA_PATH + "/part1",
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])
    )

    data_loader = torch.utils.data.DataLoader(
        custom_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True
    )

    check_data(data_loader)

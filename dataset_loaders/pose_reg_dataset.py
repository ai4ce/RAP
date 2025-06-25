"""Abstract class for pose regression dataset."""

from abc import ABC, abstractmethod

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class PoseRegDataset(ABC, Dataset):
    """Abstract class for pose regression dataset."""

    def __init__(self, train, hw, hw_gs):
        super().__init__()
        self.resize = transforms.Resize(hw)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train = train
        self.resize_gs = transforms.Resize(hw_gs)

    @property
    @abstractmethod
    def poses(self):
        """Return the poses of the dataset."""
        pass

    @property
    @abstractmethod
    def image_paths(self):
        """Return the image paths of the dataset."""
        pass

    @property
    @abstractmethod
    def image_names(self):
        """Return the image paths of the dataset."""
        pass

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.poses.shape[0]

    def __getitem__(self, index):
        """Get the sample and target at the given index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: Sample and target.
        """
        img_orig = Image.open(self.image_paths[index])  # chess img.size = (640,480)
        img_resized = self.to_tensor(self.resize(img_orig))
        img_gs = self.to_tensor(self.resize_gs(img_orig))
        img_normed = self.normalize(img_resized)
        pose = self.poses[index]
        return img_normed, pose, img_gs, self.image_names[index]

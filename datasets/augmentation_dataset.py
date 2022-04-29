import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np


class AugmentationDataset(Dataset):
    """
    Class to combine two datasets and keep track of which of the two dataset the element was sampled.
    Each sample of the dataset contains the original image as well as the augmented versions of that image.
    """
    def __init__(self, dataset1, dataset2=None, d=1, r=4, device='cpu'):
        self.dataset1 = dataset1
        self.len_dataset1 = len(dataset1)
        self.dataset2 = None
        self.d = d
        self.r = r
        self.device = device
        self.translation_vector = self.create_translation_tensors(self.d)
        self.translation_vector = self.translation_vector.to(self.device)
        self.rotation_vector = self.create_rotation_tensors(self.r)
        self.rotation_vector = self.rotation_vector.to(self.device)
        if dataset2 is not None:
            self.dataset2 = dataset2
            self.len_dataset2 = len(dataset2)

    def __len__(self):
        if self.dataset2 is None:
            return len(self.dataset1)

        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < self.len_dataset1:
            data_point = (self.dataset1[idx][0], self.dataset1[idx][1], 0)
        elif idx >= self.len_dataset1:
            data_point = (self.dataset2[idx - self.len_dataset1][0], self.dataset2[idx - self.len_dataset1][1], 1)
        else:
            raise IndexError("Given index is out of range")

        return self.augment_images(data_point[0].unsqueeze(0), self.rotation_vector, self.translation_vector).squeeze(
            1), data_point[1], data_point[2]

    @staticmethod
    def rotate_image(image: torch.Tensor, rotation_vector: torch.Tensor):
        image_vec = image.repeat(rotation_vector.shape[0], 1, 1, 1)
        radians_vec = rotation_vector * np.pi / 180
        rotation_matrices = []
        for angle in radians_vec:
            rotation_matrices.append(
                torch.tensor([
                    [torch.cos(angle), -torch.sin(angle), 0],
                    [torch.sin(angle), torch.cos(angle), 0],
                ])
            )
        rotation_matrices = torch.stack(rotation_matrices)

        grid = F.affine_grid(rotation_matrices, image_vec.size(), align_corners=True)
        grid = grid.to(image_vec.device)

        return F.grid_sample(image_vec, grid.float(), align_corners=True)

    @staticmethod
    def shift_image(image: torch.Tensor, displacement_vector: torch.Tensor):
        image_vec = image.repeat(displacement_vector.shape[0], 1, 1, 1)
        transl_matrices = []
        for displacement in displacement_vector:
            transl_matrices.append(torch.tensor([[1, 0, displacement[-2]], [0, 1, displacement[-1]]]))
        transl_matrices = torch.stack(transl_matrices).float()

        grid = F.affine_grid(transl_matrices, image_vec.size(), align_corners=True)
        grid = grid.to(image_vec.device)
        return F.grid_sample(image_vec, grid.float(), align_corners=True)

    def augment_images(self, image: torch.Tensor, rotation_vector: torch.Tensor, displacement_vector: torch.Tensor):
        image = image.to(self.device)
        rotation_vector = rotation_vector.to(self.device)
        displacement_vector = displacement_vector.to(self.device)
        rotated_images = AugmentationDataset.rotate_image(image, rotation_vector)

        displaced_images = AugmentationDataset.shift_image(image, displacement_vector)

        return torch.cat([rotated_images, displaced_images])

    @staticmethod
    def create_rotation_tensors(max_rotation=4):
        """
        Creates a vector of angles in degree for -/+ max_rotation degrees.
        The vector has 2*r+1 elements which means that each image that is going to be augmented is going to be rotated 2*r+1 times with
        steps of 1 degree.
        Use scipy.ndimage.interpolation.rotate to rotate the image.

        See https://github.com/label-only/membership-inference/blob/main/utils.py for more info.

        :param max_rotation: The maximum angle in degree by which the images are rotated.
        :return: The vector with all the rotations.
        """
        if max_rotation == 1 or max_rotation == 0:
            return torch.tensor(0.0)

        return torch.linspace(-max_rotation, max_rotation, (max_rotation * 2 + 1))

    @staticmethod
    def create_translation_tensors(max_pixel_displacement=1):
        """
        Creates a vector of translations.
        The vector has 4*d+1 elements which means that each image that is going to be augmented is going to be shifted 4*d+1 times.
        Use scipy.ndimage.interpolation.shift to shift the image.

        :param max_pixel_displacement: The maximum amount of displacement of horizontal and vertical shift together.
        """

        # create displacement vectors in vertical and horizontal direction for each combination that satisfies
        # vertical_displacement + horizontal_displacement = max_pixel_displacement
        def all_shifts(mshift):
            """Function taken from https://github.com/label-only/membership-inference/blob/main/utils.py"""
            # If there is no displacement
            if mshift == 0:
                return [[0, 0, 0, 0]]
            all_pairs = []
            start = [0, 0, mshift, 0]
            end = [0, 0, mshift, 0]
            vdir = -1
            hdir = -1
            first_time = True
            while (start[2] != end[2] or start[3] != end[3]) or first_time:
                all_pairs.append(start)
                start = [0, 0, start[2] + vdir, start[3] + hdir]
                if abs(start[2]) == mshift:
                    vdir *= -1
                if abs(start[3]) == mshift:
                    hdir *= -1
                first_time = False
            all_pairs = [[0, 0, 0, 0]] + all_pairs  # add no shift
            return all_pairs

        translates = all_shifts(max_pixel_displacement)
        return torch.tensor(translates)

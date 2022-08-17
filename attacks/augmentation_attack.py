from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from rtpt import RTPT

from .attack import PredictionScoreAttack
from utils.training import EarlyStopper


class BaseMetric():
    def __init__(self, name):
        self._num_corrects = 0
        self._num_samples = 0
        self.name = name
        super().__init__()

    def reset(self):
        self._num_corrects = 0
        self._num_samples = 0

    def update(self, model_output, y_true):
        y_pred = torch.argmax(model_output, dim=1)
        self._num_corrects += torch.sum(y_pred == y_true).item()
        self._num_samples += y_true.shape[0]

    @abstractmethod
    def compute_metric(self):
        pass


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
            1), data_point[1], data_point[2]    # label (0-9), membership status (0-1)

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
            # divide the x and y translation through height/2 and width/2 of the image
            # this is because grid_sample expects values in the range of [-1,1] (the coordinate [0,0] is the center of
            # the image while [-1, -1] is the upper left pixel)
            transl_matrices.append(
                torch.tensor(
                    [
                        [1, 0, displacement[-2]/(image.shape[-2]/2)],
                        [0, 1, displacement[-1]/(image.shape[-1]/2)]
                    ]
                )
            )
            #transl_matrices.append(torch.tensor([[1, 0, displacement[-2]], [0, 1, displacement[-1]]]))
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


class BinaryClassifierAccuracy(BaseMetric):
    def __init__(self, threshold=0.5, name='binary_accuracy'):
        super().__init__(name)

    def compute_metric(self):
        return self._num_corrects / self._num_samples

    def update(self, model_output, y_true):
        y_pred = torch.round((model_output >= 0.5).float()).squeeze()
        y_true = torch.round(y_true.float()).squeeze()
        self._num_corrects += (y_pred == y_true).sum().item()
        self._num_samples += len(y_true)


class AugmentationAttack(PredictionScoreAttack):
    def __init__(
        self,
        apply_softmax: bool,
        d=1,
        r=4,
        augmentation_batch_size: int = 16,
        batch_size: int = 64,
        log_training: bool = False
    ):
        """
        https://arxiv.org/abs/2007.14321
        :param d: The distance by which the images are shifted
        :param r: The angle by which the images are rotated
        """
        super().__init__('AugmentationAttack')

        self.apply_softmax = apply_softmax
        self.d = d
        self.r = r
        self.augmentation_batch_size = augmentation_batch_size
        self.batch_size = batch_size
        self.log_training = log_training
        self.num_augmented_images = (self.r * 2 + 1) + (self.d * 4 + 1)
        self.attack_model = nn.Sequential(nn.Linear(self.num_augmented_images, 64), nn.ReLU(), nn.Linear(64, 1))

    def get_out_features(self, model):
        try:
            out = model.linear.out_features
        except Exception as e:
            try:
                out = model.fc.out_features # resnet50
            except Exception as e:
                try:
                    out = model.model.linear.out_features #llla
                except Exception as e:
                    print(model)
        return out
        
    def learn_attack_parameters(self, shadow_model: nn.Module, member_dataset: Dataset, non_member_dataset: Dataset):
        shadow_model.to(self.device)
        shadow_model.eval()

        # combine the datasets of the shadow model
        combined_shadow_dataset = AugmentationDataset(non_member_dataset, member_dataset, self.d, self.r)
        data_loader = DataLoader(
            combined_shadow_dataset, shuffle=True, batch_size=self.augmentation_batch_size, num_workers=8
        )
        rtpt = RTPT(name_initials='BB', experiment_name='AugmentationAttack_augment_images', max_iterations=len(combined_shadow_dataset))
        rtpt.start()
        # get the prediction values of the shadow model
        with torch.no_grad():
            prediction_vectors = []
            membership_label_vector = []
            for augmented_images, class_label, no_member in tqdm(data_loader, desc="Augmenting images", leave=False, disable=not self.log_training):
                # move the images to the same device as the membership predictor
                # tensor size [224, 3, 32, 32]
                image_tensor = augmented_images.to(self.device).squeeze().view(
                    -1, augmented_images.size()[-3], augmented_images.size()[-2], augmented_images.size()[-1]
                )
                # tensor size [16,1] (values 0-9)
                class_label = class_label.to(self.device).unsqueeze(1)
                # tensor size [16,1] (values 0-1)
                no_member = no_member.to(self.device).unsqueeze(1)
                # tensor size [16, 14, 10]                
                logits = shadow_model(image_tensor).view(-1, self.num_augmented_images, self.get_out_features(shadow_model))
                output = logits.softmax(2)
                # tensor size [16, 14]
                predictions = torch.argmax(output, dim=2)
                # tensor size [16, 14] (values False,True)
                correct_class_predictions = (predictions == class_label)

                # list with tensor size [16, 14]
                prediction_vectors.append(correct_class_predictions)
                # list with tensor size [16, 1]
                membership_label_vector.append(no_member)
                rtpt.step()

        # tensor size [5000, 14] (values True, False): is class correctly predicted
        prediction_vectors = torch.cat(prediction_vectors)
        # tensor size [5000, 1] (values 0, 1)
        membership_label_vector = torch.cat(membership_label_vector)

        membership_predictor_dataloader = DataLoader(
            TensorDataset(prediction_vectors, membership_label_vector), batch_size=self.batch_size
        )
        optimizer = torch.optim.Adam(self.attack_model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        accuracy = BinaryClassifierAccuracy()

        # train the membership predictor network on the combined dataset by iterating
        # each original image and getting the augmented images of this image back
        self.attack_model.train()
        self.attack_model.to(self.device)
        early_stopper = EarlyStopper(window=15, min_diff=0.0005)
        epoch = 0
        last_loss = early_stopper.best_value
        best_acc = 0
        best_model_state_dict = None
        while not early_stopper.stop_early(last_loss):
            running_loss = 0.0
            accuracy.reset()
            rtpt = RTPT(name_initials='BB', experiment_name='AugmentationAttack_attackmodel', max_iterations=len(membership_predictor_dataloader))
            rtpt.start()
            for idx, (batch, member_labels) in enumerate(membership_predictor_dataloader):
                # batch: [128, 14] (False, True), member_labels: [128, 1] (0,1)
                batch, member_labels = batch.to(self.device), member_labels.to(self.device)
                optimizer.zero_grad()
                # [128, 1] (-0.xxxx)
                outputs = self.attack_model(batch.float())

                loss = criterion(outputs, member_labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch.size(0)
                accuracy.update(outputs.sigmoid(), member_labels)
                rtpt.step()

            training_loss = running_loss / len(membership_predictor_dataloader)
            last_loss = training_loss
            epoch += 1

            if best_acc < accuracy.compute_metric():
                best_model_state_dict = self.attack_model.state_dict()
                best_acc = accuracy.compute_metric()

            if self.log_training:
                print(
                    f'Epoch {epoch + 1} \t Loss: {training_loss:.4f} \t Training Accuracy: {accuracy.compute_metric():.4f}'
                )

        if self.log_training:
            print(f'Using best model with training accuracy of {best_acc:.4f}')
        self.attack_model.load_state_dict(best_model_state_dict)

    def predict_membership(self, target_model: nn.Module, dataset: Dataset,member) -> np.ndarray:
        predictions = self.get_attack_model_prediction_scores(target_model, dataset, member)

        return (predictions >= 0.5).numpy()

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset,member) -> torch.Tensor:
        self.attack_model.eval()

        prediction_vectors = self._get_prediction_vectors(target_model, dataset) # [2500,14]
        dataloader = DataLoader(TensorDataset(prediction_vectors), batch_size=self.batch_size) # batch size 128

        with torch.no_grad():
            membership_preds = [] 
            for vec in dataloader:  # [128, 14] x 20
                output = self.attack_model(vec[0].float()) # [1, 128]
                membership_preds.append(output.sigmoid()) # [128, 1]

        predictions = torch.cat(membership_preds, dim=0).squeeze() # [2500]

        return predictions.cpu()

    def _get_prediction_vectors(self, target_model: nn.Module, dataset: Dataset):
        target_model.eval()

        augmentation_dataset = AugmentationDataset(dataset, d=self.d, r=self.r)
        data_loader = DataLoader(
            augmentation_dataset, shuffle=True, batch_size=self.augmentation_batch_size, num_workers=8
        )
        rtpt = RTPT(name_initials='BB', experiment_name='AugmentationAttack_prediction_vecs', max_iterations=len(augmentation_dataset))
        rtpt.start()
        # get the prediction values of the shadow model
        with torch.no_grad():
            prediction_vectors = []
            for augmented_images, class_label, _ in tqdm(data_loader, desc="Augmenting images", leave=False, disable=not self.log_training):
                # move the images to the same device as the membership predictor
                image_tensor = augmented_images.to(self.device).squeeze().view(
                    -1, augmented_images.size()[-3], augmented_images.size()[-2], augmented_images.size()[-1]
                )
                class_label = class_label.to(self.device).unsqueeze(1)
                logits = target_model(image_tensor
                                      ).view(-1, self.num_augmented_images, self.get_out_features(target_model))
                output = logits.softmax(2)
                predictions = torch.argmax(output, dim=2)
                correct_class_predictions = (predictions == class_label)

                prediction_vectors.append(correct_class_predictions)
                rtpt.step()

        prediction_vectors = torch.cat(prediction_vectors)

        return prediction_vectors

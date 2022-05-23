from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from .attack import PredictionScoreAttack
from utils.training import EarlyStopper


class GapAttack(PredictionScoreAttack):
    def __init__(
        self, 
        apply_softmax: bool,
        batch_size: int = 128,
        log_training: bool = False
    ):
        """
        https://arxiv.org/abs/2007.14321
        """
        super().__init__('GapAttack')

        self.apply_softmax = apply_softmax
        self.batch_size = batch_size
        self.log_training = log_training
        
    def learn_attack_parameters(self, shadow_model: nn.Module, member_dataset: Dataset, non_member_dataset: Dataset):
        # nothing to do, as this is a baseline attack which predicts any misclassified point as non-member
        pass
    
    def predict_membership(self, target_model: nn.Module, dataset: Dataset) -> np.ndarray:
        """
        Predicts for samples X if they were part of the training set of the target model.
        Returns True if membership is predicted, False else.
        """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8)
        predictions = []
        target_model.eval()
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            with torch.no_grad():
                y_pred = torch.argmax(target_model.forward(X), dim=1)
                predictions.append(y_pred == y)
        return np.array(torch.cat(predictions).cpu().tolist())

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        predictions = self.predict_membership(target_model, dataset)
        return torch.from_numpy(np.array(predictions)*1)
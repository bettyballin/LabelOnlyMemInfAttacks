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
        self.attack_model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
        
    def learn_attack_parameters(self, shadow_model: nn.Module, member_dataset: Dataset, non_member_dataset: Dataset):
        pass
    
    def predict_membership(self, target_model: nn.Module, dataset: Dataset) -> np.ndarray:
        """
        Predicts for samples X if they were part of the training set of the target model.
        Returns True if membership is predicted, False else.
        """
        predictions = self.get_attack_model_prediction_scores(target_model, dataset)
        return predictions.numpy()

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8)
        predictions = []
        target_model.eval()
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            with torch.no_grad():
                y_pred = torch.argmax(shadow_model.forward(X), dim=1)
                predictions.append(y_pred == y)
        return torch.cat(predictions).cpu().tolist()
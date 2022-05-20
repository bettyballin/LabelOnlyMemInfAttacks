from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Tuple

from .attack import PredictionScoreAttack
from .hopskipjump import HopSkipJump
from utils.training import EarlyStopper

class DecisionBoundaryAttack(PredictionScoreAttack):
    def __init__(
        self,
        apply_softmax: bool,
        batch_size: int = 128,
        log_training: bool = False,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        tau: float = 0.5,
        max_iter: int = 10,
        max_eval: int = 500,
        init_eval: int = 10,
        init_size: int = 100,
    ):
        """
        Create a `LabelOnlyDecisionBoundary` instance for Label-Only Inference Attack based on Decision Boundary.
        :param estimator: A trained classification estimator.
        :param distance_threshold_tau: Threshold distance for decision boundary. Samples with boundary distances larger
                                       than threshold are considered members of the training dataset.
        """
        super().__init__('DecisionBoundaryAttack')

       
        self.batch_size = batch_size
        self.tau = tau
        self.input_shape = input_shape
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.init_size = init_size
        self.apply_softmax = apply_softmax
        self.log_training = log_training

    def learn_attack_parameters(
        self, shadow_model: nn.Module, member_dataset: torch.utils.data.Dataset, non_member_dataset: Dataset, *kwargs
    ):
        """
        Calibrate distance threshold maximizing the membership inference accuracy on `x_train` and `x_test`.
        Keyword Arguments for HopSkipJump:
            * *norm*: Order of the norm. Possible values: "inf", np.inf or 2.
            * *max_iter*: Maximum number of iterations.
            * *max_eval*: Maximum number of evaluations for estimating gradient.
            * *init_eval*: Initial number of evaluations for estimating gradient.
            * *init_size*: Maximum number of trials for initial generation of adversarial examples.
        """
        shadow_model.to(self.device)
        shadow_model.eval()

        hsj = HopSkipJump(classifier=shadow_model, apply_softmax=self.apply_softmax, input_shape=self.input_shape, device=self.device)

        with torch.no_grad():
            distance_train = []
            distance_test = []
            for i, dataset in enumerate([non_member_dataset, member_dataset]):
                loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8)
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    x_adv = hsj.generate(x=x, y=y)
                    #print(np.array(x_adv).shape) [128,3,32,32]
                    #x_adv = np.load(f)
                    output = shadow_model(x) 
                    if self.apply_softmax:
                        output = output.softmax(dim=1)
                    y_pred = torch.argmax(output, dim=1)
                    x, y_pred, y = x.cpu().numpy(), y_pred.cpu().numpy(), y.cpu().numpy()
                    distance = np.linalg.norm((x_adv - x).reshape((x.shape[0], -1)), ord=2, axis=1) # [batchsize]
                    distance[y_pred != y] = 0
                    if i == 0:
                        distance_train.append(np.amax(distance))
                    else:
                        distance_test.append(np.amax(distance))
            tau_increment = np.amax([np.amax(distance_train), np.amax(distance_test)]) / 100
            acc_max = 0.0
            distance_threshold_tau = 0.0

        for i_tau in range(1, 100):

            is_member_train = np.where(distance_train > i_tau * tau_increment, 1, 0)
            is_member_test = np.where(distance_test > i_tau * tau_increment, 1, 0)

            acc = (np.sum(is_member_train) + (is_member_test.shape[0] - np.sum(is_member_test))
                   ) / (is_member_train.shape[0] + is_member_test.shape[0])

            if acc > acc_max:
                distance_threshold_tau = i_tau * tau_increment
                acc_max = acc

        if self.log_training:
            print(f'Setting threshold to {distance_threshold_tau}')

        self.tau = distance_threshold_tau

    def predict_membership(self, target_model: nn.Module, dataset: Dataset):
         return self.get_attack_model_prediction_scores(target_model, dataset) == 1

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        """
        Infer membership of input `x` in estimator's training data.
        :Keyword Arguments for HopSkipJump:
            * *norm*: Order of the norm. Possible values: "inf", np.inf or 2.
            * *max_iter*: Maximum number of iterations.
            * *max_eval*: Maximum number of evaluations for estimating gradient.
            * *init_eval*: Initial number of evaluations for estimating gradient.
            * *init_size*: Maximum number of trials for initial generation of adversarial examples.
            * *verbose*: Show progress bars.
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member.
        """
        hsj = HopSkipJump(classifier=target_model, apply_softmax=self.apply_softmax, input_shape=self.input_shape, device=self.device)
        dist = []
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8)
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                x_adv = hsj.generate(x=x, y=y)
                output = target_model(x)
                if self.apply_softmax:
                    output = output.softmax(dim=1)
                y_pred = torch.argmax(output, dim=1)
                x, y, y_pred = x.cpu().numpy(), y.cpu().numpy(), y_pred.cpu().numpy()
                distance = np.linalg.norm((x_adv - x).reshape((x.shape[0], -1)), ord=2, axis=1)
                distance[y_pred != y] = 0
                dist.append(np.where(distance > self.tau, 1, 0))
        is_member = np.array(dist).reshape(-1)
        return torch.from_numpy(is_member)   
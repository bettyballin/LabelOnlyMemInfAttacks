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
        input_shape: Tuple[int, int, int] = (3, 224, 224),
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
        # get the membership and non-membership data as numpy arrays
        x_train = next(iter(torch.utils.data.DataLoader(member_dataset, batch_size=len(member_dataset))))[0].numpy()
        y_train = next(iter(torch.utils.data.DataLoader(member_dataset, batch_size=len(member_dataset))))[1].numpy()
        x_test = next(iter(torch.utils.data.DataLoader(non_member_dataset,
                                                       batch_size=len(non_member_dataset))))[0].numpy()
        y_test = next(iter(torch.utils.data.DataLoader(non_member_dataset,
                                                       batch_size=len(non_member_dataset))))[1].numpy()

        hsj = HopSkipJump(classifier=shadow_model, input_shape=self.input_shape)

        x_train_adv = hsj.generate(x=x_train, y=y_train)
        x_test_adv = hsj.generate(x=x_test, y=y_test)

        distance_train = np.linalg.norm((x_train_adv - x_train).reshape((x_train.shape[0], -1)), ord=2, axis=1)
        distance_test = np.linalg.norm((x_test_adv - x_test).reshape((x_test.shape[0], -1)), ord=2, axis=1)

        y_train_pred = shadow_model.predict(x_train, numpy=True)
        y_test_pred = shadow_model.predict(x_test, numpy=True)

        distance_train[y_train_pred != y_train] = 0
        distance_test[y_test_pred != y_test] = 0

        num_increments = 100
        tau_increment = np.amax([np.amax(distance_train), np.amax(distance_test)]) / num_increments

        acc_max = 0.0
        distance_threshold_tau = 0.0

        for i_tau in range(1, num_increments):

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
        x = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset))))[0].numpy()
        y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset))))[1].numpy()

        hsj = HopSkipJump(classifier=target_model, input_shape=self.input_shape)
        x_adv = hsj.generate(x=x, y=y)
        distance = np.linalg.norm((x_adv - x).reshape((x.shape[0], -1)), ord=2, axis=1)
        output = target_model(x).softmax(dim=1)
        y_pred = torch.argmax(output, dim=1)
        distance[y_pred != y] = 0

        is_member = np.where(distance > self.tau, 1, 0)

        return torch.tensor(is_member)   

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
        return self.predict_membership(self, target_model, dataset)

from abc import abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from rtpt import RTPT

from .attack import PredictionScoreAttack
from utils.training import EarlyStopper

class RandomNoiseAttack(PredictionScoreAttack):
    """
        Target model is the model targeted by the adversarial.
        Shadow model corresponds to the local source model trained by the adversarial to fit the attack.
        N refers to the number of inferences per attack (number of perturbed images per input sample).
        Sigma and tau refer to the Gaussian noise's std and decision threshold, respectively.
        https://arxiv.org/abs/2007.14321

        :param N: The number of perturbations per input sample.
        :param tau: The threshold which is going to be estimated at which the given samples are predicted as member samples.
        :param min_sigma:   The minimum sigma which is used to create a range of sigma values to try out when estimating the distance
                            to the decision boundary.
        :param max_sigma:   The maximum sigma which is used to create a range of sigma values to try out when estimating the distance
                            to the decision boundary.
        :param num_sigmas: The number of sigmas that are generated between min_sigma and max_sigma.
        :param batch_size:  The batch size that is used to predict tau. Careful, since for each sample N perturbations are created. As
                            a result increasing this value might result in a high memory usage.
    """
    def __init__(
        self, 
        apply_softmax: bool,
        N=2500,
        tau=0,
        min_sigma=0,
        max_sigma=1,
        num_sigmas=20,
        early_stopping=3,
        batch_size: int = 128,
        log_training: bool = False
        ):
        super().__init__('RandomNoiseAttack')

        self.apply_softmax = apply_softmax
        self.batch_size = batch_size
        self.log_training = log_training
        self.tau = tau
        self.sigma = min_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigmas = num_sigmas
        self.N = N
        self.early_stopping = early_stopping        

    def estimate_tau(self, shadow_model: nn.Module, member_dataset: Dataset, non_member_dataset: Dataset, sigma):
        """
         Estimate tau for a given sigma on the shadow model.
         First, the distance for membership and non-membership samples is computed on the shadow model.
         Distance is approximated using the model's accuracy under perturbation.
         Then, a simple linear search is performed to find decision threshold tau.
         Tau is chosen to maximize the accuracy on the membership estimation.
        """
        shadow_model.to(self.device)
        shadow_model.eval()
        
        membership_loader = DataLoader(member_dataset, shuffle=True, batch_size=self.batch_size, num_workers=8)
        non_membership_loader = DataLoader(non_member_dataset, shuffle=True, batch_size=self.batch_size, num_workers=8)
                
        # estimate distance of decision boundary
        member_distances = np.empty(len(member_dataset))
        non_member_distances = np.empty(len(non_member_dataset))
        rtpt = RTPT(name_initials='BB', experiment_name='RandomNoise_estimate_tau', max_iterations=len(member_dataset)+len(non_member_dataset))
        rtpt.start()
        for i, (x, y) in enumerate(tqdm(membership_loader, desc='Estimating distance for membership samples')):
            distances = self.estimate_distance(x, y, shadow_model, sigma)
            member_distances[i * len(x):i * len(x) + len(x)] = distances
            rtpt.step()
        for i, (x, y) in enumerate(tqdm(non_membership_loader, desc='Estimating distance for non-membership samples')):
            distances = self.estimate_distance(x, y, shadow_model, sigma)
            non_member_distances[i * len(x):i * len(x) + len(x)] = distances
            rtpt.step()

        # estimate membership decision boundary tau by linear search
        best_acc = 0.0
        tau = 0.0
        for thresh in np.linspace(0, 1, 10000):
            acc = (np.sum(member_distances > thresh) + np.sum(non_member_distances <= thresh)) \
                  / (len(member_dataset) + len(non_member_dataset))
            if acc > best_acc:
                best_acc = acc
                tau = thresh
        return tau, best_acc

    def learn_attack_parameters(self, shadow_model: nn.Module, member_dataset: Dataset, non_member_dataset: Dataset):
        """
        Estimates standard deviation for isotropic Gaussian noise and threshold tau.
        A grid search on all sigma values is performed based on the best resulting accuracy.
        The method also estimates an apropiate decision threshold tau.
        The shadow model is used to tune the parameters.
        """
        if self.log_training:
            print(f'Estimating parameters on {len(member_dataset)} membership and {len(non_member_dataset)} non-membership samples with {self.N} perturbations per sample')
        
        sigma_values = [i for i in np.linspace(self.min_sigma, self.max_sigma, self.num_sigmas)]
        best_sigma = 0
        best_tau = 0
        best_acc = 0
        no_improvements = 0
        for sigma in sigma_values:
            if sigma == 0:
                continue
            tau, acc = self.estimate_tau(shadow_model, member_dataset, non_member_dataset, sigma)
            if acc > best_acc:
                best_acc = acc
                best_sigma = sigma
                best_tau = tau
                no_improvements = 0
            else:
                no_improvements += 1

            if self.log_training:
                print(f'sigma={sigma:.4f} tau={tau:.4f} acc={acc:.4f} - best acc={best_acc:.4f}')
            
            if no_improvements >= self.early_stopping:
                if self.log_training:
                    print('Early stopping parameter search')
                break
            
        self.tau = best_tau
        self.sigma = best_sigma
        if self.log_training:
            print(f'Parameters estimated: sigma={best_sigma:.4f}, tau={best_tau:.4f}. Achieved accuracy={best_acc:.4f}')

    def estimate_distance(self, x, y, shadow_model, sigma):
        """
        Distance estimated by computing the accuracy of the model
        on N perturbed samples using an isotropic Gaussian noise.
        Note: x refers to a single sample image.
        """
        shadow_model.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            X_noisy = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1)
            X_noisy = X_noisy + torch.randn_like(X_noisy) * sigma
            X_noisy = X_noisy.view(-1, X_noisy.size()[-3], X_noisy.size()[-2], X_noisy.size()[-1]) # dim?
            n_batches = math.ceil(len(X_noisy) / self.batch_size)
            predictions = torch.tensor([], device=self.device)
            for i in tqdm(range(n_batches), desc='Distance estimation', leave=False, disable=not self.log_training):
                input = X_noisy[i * self.batch_size:(i + 1) * self.batch_size]
                output = shadow_model.forward(input)
                y_pred = torch.argmax(output, dim=1)
                predictions = torch.cat((predictions, y_pred))
            correct_predictions = predictions == y.repeat_interleave(self.N)
            correct_predictions = correct_predictions.view(x.size()[0], self.N)
            correct_predictions = torch.sum(correct_predictions, dim=1)
        return (correct_predictions / self.N).detach().cpu().numpy()

    def predict_membership(self, target_model: nn.Module, dataset: Dataset) -> np.ndarray:
        """
        Predicts for samples X if they were part of the training set of the target model.
        Returns True if membership is predicted, False else.
        """
        predictions = self.get_attack_model_prediction_scores(target_model, dataset)
        return predictions.numpy() == 1

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        target_model.eval()
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size, num_workers=8)
        predictions = np.zeros(len(dataset), dtype=bool)
        rtpt = RTPT(name_initials='BB', experiment_name='DecisionBoundary_predict', max_iterations=len(dataset))
        rtpt.start()
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                output = target_model(x)
                if self.apply_softmax:
                    output = output.softmax(dim=1)
                y_pred = torch.argmax(output, dim=1)
                dist = self.estimate_distance(x, y, target_model, self.sigma)

                # Set distance to 0 for false predictions.
                mask = (y_pred == y).cpu().numpy()
                dist[mask == False] = 0

                predictions[i * len(x):i * len(x) + len(x)] = dist > self.tau
                rtpt.step()

        return torch.from_numpy(predictions*1)
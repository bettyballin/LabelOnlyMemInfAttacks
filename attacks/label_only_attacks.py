import math
from typing import Tuple
import numpy as np
import torch

from attacks.hopskipjump import HopSkipJump
from attacks.attack import MembershipInferenceAttack
from datasets.augmentation_dataset import AugmentationDataset
from metrics.accuracy import BinaryClassifierAccuracy
from models.membership_predictor import MembershipPredictor
#from utils.configs import TrainConfigHelper
#from utils.wandb_utils import get_shadow_model
from utils.training import EarlyStopper

from rtpt import RTPT
from torch.utils.data import dataloader
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm

from utils.dataset_utils import get_subsampled_dataset


class RandomNoiseAttack(MembershipInferenceAttack):
    def __init__(self, target_model, rtpt: RTPT, mix_validation_and_train_set=False):
        """
        Target model is the model targeted by the adversarial.
        Shadow model corresponds to the local source model trained by the adversarial to fit the attack.
        N refers to the number of inferences per attack (number of perturbed images per input sample).
        Sigma and tau refer to the Gaussian noise's std and decision threshold, respectively.
        https://arxiv.org/abs/2007.14321
        """
        super().__init__(target_model, 'RandomNoiseAttack', rtpt, mix_validation_and_train_set)

    def setup_attack(
        self,
        train_config_helper: TrainConfigHelper,
        dataset_size,
        use_shadow_model=True,
        shadow_model_run_path="",
        N=2500,
        tau=0,
        min_sigma=0,
        max_sigma=1,
        num_sigmas=20,
        batch_size=128,
        early_stopping_param_search=5,
        relabel_shadow_dataset=False,
        early_stopping_relabelled_shadow_training={
            "window": 5, "min_diff": 0.005
        },
        retrain_relabelled_shadow_model=True
    ):
        """
        :param train_config_helper: The config that was used to train the model.
        :param dataset_size: The size of the member and non-member dataset, respectively.
        :param use_shadow_model: Whether the shadow model is used for the attack. If this parameter is set to false,
            the attack is directly applied to the target model.
        :param shadow_model_run_path: The given shadow model run path.
        :param N: The number of perturbations per input sample.
        :param tau: The threshold which is going to be estimated at which the given samples are predicted as member samples.
        :param min_sigma:   The minimum sigma which is used to create a range of sigma values to try out when estimating the distance
                            to the decision boundary.
        :param max_sigma:   The maximum sigma which is used to create a range of sigma values to try out when estimating the distance
                            to the decision boundary.
        :param num_sigmas: The number of sigmas that are generated between min_sigma and max_sigma.
        :param batch_size:  The batch size that is used to predict tau. Careful, since for each sample N perturbations are created. As
                            a result increasing this value might result in a high memory usage.
        :param early_stopping_param_search: If there is no improvement in accuracy while estimating tau, the search is stopped early.
        """
        self.tau = tau
        self.sigma = min_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigmas = num_sigmas
        self.N = N
        self.batch_size = batch_size
        self.relabel_shadow_dataset = relabel_shadow_dataset
        self.early_stopping_relabelled_shadow_training = early_stopping_relabelled_shadow_training

        # Set seed
        seed = train_config_helper.config.seed
        torch.manual_seed(seed)

        # get the dataset and split in in half
        train_set, valid_set, test_set = train_config_helper.get_pytorch_dataset(
            train_config_helper.dataset_config.args, seed=seed, shadow_datasets=True)

        self.target_train_set, self.shadow_train_set = train_set
        self.target_valid_set, self.shadow_valid_set = valid_set
        self.target_test_set, self.shadow_test_set = test_set

        self.attacked_model = self.target_model
        if use_shadow_model:
            self.attacked_model = get_shadow_model(
                self.shadow_train_set,
                self.shadow_valid_set,
                self.target_model,
                shadow_model_run_path,
                relabel_shadow_dataset,
                early_stopping_relabelled_shadow_training,
                retrain_relabelled_shadow_model,
                self.batch_size,
                self.rtpt
            )

        self.target_membership_set, self.target_non_membership_set = self.get_membership_splits(
            self.target_train_set,
            self.target_valid_set,
            self.target_test_set,
            dataset_size, seed,
            self.mix_validation_and_train_set
        )
        self.shadow_membership_set, self.shadow_non_membership_set = self.get_membership_splits(
            self.shadow_train_set,
            self.shadow_valid_set,
            self.shadow_test_set,
            dataset_size, seed,
            self.mix_validation_and_train_set
        )

        sigma_values = [i for i in np.linspace(min_sigma, max_sigma, num_sigmas)]
        self.rtpt.max_iterations = len(sigma_values) + len(self.target_membership_set
                                                           ) + len(self.target_non_membership_set)
        self.estimate_parameters(
            sigma_values=sigma_values,
            early_stopping=early_stopping_param_search,
            membership_data=self.shadow_membership_set,
            non_membership_data=self.shadow_non_membership_set
        )

        return self.target_membership_set, self.target_non_membership_set

    def estimate_parameters(self, sigma_values, membership_data, non_membership_data, early_stopping=3):
        """
        Estimates standard deviation for isotropic Gaussian noise and threshold tau.
        A grid search on all sigma values is performed based on the best resulting accuracy.
        The method also estimates an apropiate decision threshold tau.
        The shadow model is used to tune the parameters.
        """
        print(
            f'Estimating parameters on {len(membership_data)} membership and {len(non_membership_data)} non-membership samples with {self.N} perturbations per sample'
        )
        best_sigma = 0
        best_tau = 0
        best_acc = 0
        no_improvements = 0
        for sigma in sigma_values:
            if sigma == 0:
                continue
            tau, acc = self.estimate_tau(membership_data, non_membership_data, sigma)
            if acc > best_acc:
                best_acc = acc
                best_sigma = sigma
                best_tau = tau
                no_improvements = 0
            else:
                no_improvements += 1
            print(f'sigma={sigma:.4f} tau={tau:.4f} acc={acc:.4f} - best acc={best_acc:.4f}')
            if no_improvements >= early_stopping:
                print('Early stopping parameter search')
                break
            if self.rtpt is not None:
                self.rtpt.step()
        self.tau = best_tau
        self.sigma = best_sigma
        print(f'Parameters estimated: sigma={best_sigma:.4f}, tau={best_tau:.4f}. Achieved accuracy={best_acc:.4f}')

    def estimate_tau(self, membership_data, non_membership_data, sigma):
        """
         Estimate tau for a given sigma on the shadow model.
         First, the distance for membership and non-membership samples is computed on the shadow model.
         Distance is approximated using the model's accuracy under perturbation.
         Then, a simple linear search is performed to find decision threshold tau.
         Tau is chosen to maximize the accuracy on the membership estimation.
        """
        # estimate distance of decision boundary
        membership_loader = DataLoader(membership_data, batch_size=self.batch_size, num_workers=4)
        non_membership_loader = DataLoader(non_membership_data, batch_size=self.batch_size, num_workers=4)
        member_distances = np.empty(len(membership_data))
        non_member_distances = np.empty(len(non_membership_data))
        self.attacked_model.eval()
        for i, (x, y) in enumerate(tqdm(membership_loader, desc='Estimating distance for membership samples')):
            distances = self.estimate_distance(x, y, self.attacked_model, sigma)
            member_distances[i * len(x):i * len(x) + len(x)] = distances
        for i, (x, y) in enumerate(tqdm(non_membership_loader, desc='Estimating distance for non-membership samples')):
            distances = self.estimate_distance(x, y, self.attacked_model, sigma)
            non_member_distances[i * len(x):i * len(x) + len(x)] = distances

        # estimate membership decision boundary tau by linear search
        best_acc = 0.0
        tau = 0.0
        for thresh in np.linspace(0, 1, 10000):
            acc = (np.sum(member_distances > thresh) + np.sum(non_member_distances <= thresh)) \
                  / (len(membership_data) + len(non_membership_data))
            if acc > best_acc:
                best_acc = acc
                tau = thresh
        return tau, best_acc

    def estimate_distance(self, x, y, model, sigma):
        """
        Distance estimated by computing the accuracy of the model
        on N perturbed samples using an isotropic Gaussian noise.
        Note: x refers to a single sample image.
        """
        model.eval()
        with torch.no_grad():
            x, y = x.to(model.device), y.to(model.device)
            X_noisy = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1)
            X_noisy = X_noisy + torch.randn_like(X_noisy) * sigma
            X_noisy = X_noisy.view(-1, X_noisy.size()[-3], X_noisy.size()[-2], X_noisy.size()[-1])
            n_batches = math.ceil(len(X_noisy) / self.batch_size)
            predictions = torch.tensor([], device=model.device)
            for i in tqdm(range(n_batches), desc='Distance estimation', leave=False):
                input = X_noisy[i * self.batch_size:(i + 1) * self.batch_size]
                output = model.forward(input)
                y_pred = torch.argmax(output, dim=1)
                predictions = torch.cat((predictions, y_pred))
            correct_predictions = predictions == y.repeat_interleave(self.N)
            correct_predictions = correct_predictions.view(x.size()[0], self.N)
            correct_predictions = torch.sum(correct_predictions, dim=1)
        return (correct_predictions / self.N).detach().cpu().numpy()

    def infer(self, dataset, target_model=None, batch_size=128):
        """
        Predicts for samples X if they were part of the training set of the target model.
        Returns True if membership is predicted, False else.
        """
        if target_model is None:
            target_model = self.target_model
        target_model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
        predictions = np.zeros(len(dataset), dtype=bool)
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(target_model.device), y.to(target_model.device)
            with torch.no_grad():
                y_pred = target_model.predict(x)
                dist = self.estimate_distance(x, y, target_model, self.sigma)

                # Set distance to 0 for false predictions.
                mask = (y_pred == y).cpu().numpy()
                dist[mask == False] = 0

                predictions[i * len(x):i * len(x) + len(x)] = dist > self.tau
            if self.rtpt is not None:
                self.rtpt.step()
        return predictions

    def evaluate(self, membership_data, non_membership_data):
        """
        Compute the attack accuracy on the target model.
        """
        print(
            f'Start evaluation on {len(membership_data)} membership and {len(non_membership_data)} non-membership samples'
        )

        correct_memberships = np.sum(np.array(self.infer(membership_data)) == True)
        correct_non_memberships = np.sum(np.array(self.infer(non_membership_data)) == False)

        acc = (correct_memberships + correct_non_memberships) / \
              (len(membership_data) + len(non_membership_data))
        acc_membership = correct_memberships / len(membership_data)
        acc_non_membership = correct_non_memberships / len(non_membership_data)

        print(
            f'Evaluation result: acc={acc:.4f}, acc_membership={acc_membership:.4f}, acc_non_membership={acc_non_membership:.4f}'
        )
        return acc, acc_membership, acc_non_membership


class AugmentationAttack(MembershipInferenceAttack):
    def __init__(self, target_model, rtpt: RTPT, mix_validation_and_train_set=False):
        """
        https://arxiv.org/abs/2007.14321
        :param d: The distance by which the images are shifted
        :param r: The angle by which the images are rotated
        """
        super().__init__(target_model, 'AugmentationAttack', rtpt, mix_validation_and_train_set)

    def setup_attack(
        self,
        target_train_config_helper,
        dataset_size,
        use_shadow_model=True,
        shadow_model_run_path="",
        d=1,
        r=4,
        membership_predictor_early_stopping_window=10,
        membership_predictor_early_stopping_mind_diff=0.005,
        membership_predictor_batch_size=16,
        membership_predictor_lr=0.001,
        augmentation_batch_size=16,
        target_model_inference_batch_size=16,
        membership_predictor_hidden_layers=[10],
        data_loader_workers=16,
        relabel_shadow_dataset=False,
        early_stopping_relabelled_shadow_training={
            "window": 5, "min_diff": 0.005
        },
        retrain_relabelled_shadow_model=True
    ):
        """
        :param target_train_config_helper: The config that was used to train the target model.
        :param dataset_size: The size of the member and non-member dataset, respectively.
        :param use_shadow_model: Whether the shadow model is used for the attack. If this parameter is set to false,
            the attack is directly applied to the target model.
        :param shadow_model_run_path: The given shadow model run path.
        :param d: The distance in pixels by which the images are shifted.
        :param r: The angle by which the images are rotated.
        :param membership_predictor_early_stopping_window: The number of epochs after which training is stopped if there is no improvement.
        :param membership_predictor_early_stopping_mind_diff: The minimum difference the loss needs to be counted as an improvement.
        :param membership_predictor_batch_size: The batch size that is used to train the membership predictor.
        :param membership_predictor_lr: The learning rate that is used to train the membership predictor.
        :param data_loader_workers: The number of workers used to load data.
        """
        self.d = d
        self.r = r
        self.membership_predictor_early_stopping_window = membership_predictor_early_stopping_window
        self.membership_predictor_early_stopping_mind_diff = membership_predictor_early_stopping_mind_diff
        self.membership_predictor_batch_size = membership_predictor_batch_size
        self.membership_predictor_hidden_layers = membership_predictor_hidden_layers
        self.augmentation_batch_size = augmentation_batch_size
        self.target_model_inference_batch_size = target_model_inference_batch_size
        self.membership_predictor_lr = membership_predictor_lr
        self.number_of_augmented_images = (r * 2 + 1) + (d * 4 + 1)
        self.data_loader_workers = data_loader_workers
        self.relabel_shadow_dataset = relabel_shadow_dataset
        self.early_stopping_relabelled_shadow_training = early_stopping_relabelled_shadow_training

        # Set seed
        seed = target_train_config_helper.config.seed
        torch.manual_seed(seed)

        # get the dataset and split in in half
        train_set, valid_set, test_set = target_train_config_helper.get_pytorch_dataset(
            target_train_config_helper.dataset_config.args, seed=seed, shadow_datasets=True)

        self.target_train_set, self.shadow_train_set = train_set
        self.target_valid_set, self.shadow_valid_set = valid_set
        self.target_test_set, self.shadow_test_set = test_set

        self.attacked_model = self.target_model
        if use_shadow_model:
            self.attacked_model = get_shadow_model(
                self.shadow_train_set,
                self.shadow_valid_set,
                self.target_model,
                shadow_model_run_path,
                relabel_shadow_dataset,
                early_stopping_relabelled_shadow_training,
                retrain_relabelled_shadow_model,
                self.target_model_inference_batch_size,
                self.rtpt
            )

        self.target_membership_set, self.target_non_membership_set = self.get_membership_splits(
            self.target_train_set,
            self.target_valid_set,
            self.target_test_set,
            dataset_size, seed,
            self.mix_validation_and_train_set
        )
        self.shadow_membership_set, self.shadow_non_membership_set = self.get_membership_splits(
            self.shadow_train_set,
            self.shadow_valid_set,
            self.shadow_test_set,
            dataset_size, seed,
            self.mix_validation_and_train_set
        )

        print(
            "Target Membership Set Size:",
            len(self.target_membership_set),
            " | ",
            "Target Non-Membership Set Size:",
            len(self.target_non_membership_set)
        )
        print(
            "Shadow Membership Set Size:",
            len(self.shadow_membership_set),
            " | ",
            "Shadow Non-Membership Set Size:",
            len(self.shadow_non_membership_set)
        )

        # train the membership predictor
        self.rtpt.max_iterations = len(self.shadow_membership_set) + len(self.shadow_non_membership_set) + len(
            self.target_membership_set
        ) + len(self.target_non_membership_set)
        self.membership_predictor = self.train_membership_predictor()

        return self.target_membership_set, self.target_non_membership_set

    def train_membership_predictor(self):
        # combine the datasets of the shadow model
        combined_shadow_dataset = AugmentationDataset(
            self.shadow_non_membership_set, self.shadow_membership_set, self.d, self.r
        )
        data_loader = DataLoader(
            combined_shadow_dataset,
            shuffle=True,
            batch_size=self.augmentation_batch_size,
            num_workers=self.data_loader_workers
        )

        # get the prediction values of the shadow model
        with torch.no_grad():
            prediction_vectors = []
            membership_label_vector = []
            self.attacked_model.eval()
            for augmented_images, class_label, no_member in tqdm(data_loader, desc="Augmenting images"):
                # move the images to the same device as the membership predictor
                image_tensor = augmented_images.to(self.attacked_model.device).squeeze().view(
                    -1, augmented_images.size()[-3], augmented_images.size()[-2], augmented_images.size()[-1]
                )
                class_label = class_label.to(self.attacked_model.device).unsqueeze(1)
                no_member = no_member.to(self.attacked_model.device).unsqueeze(1)
                logits = self.attacked_model(image_tensor).view(
                    -1, self.number_of_augmented_images, self.attacked_model.num_classes
                )
                # get the argmax of the logits. Softmax is not needed since argmax(logits)==argmax(softmax)
                predictions = torch.argmax(logits, dim=2)
                correct_class_predictions = predictions == class_label

                prediction_vectors.append(correct_class_predictions)
                membership_label_vector.append(no_member)

                if self.rtpt is not None:
                    self.rtpt.step()

        # create everything we need to train the membership predictor
        membership_predictor = MembershipPredictor(
            self.number_of_augmented_images, self.membership_predictor_hidden_layers
        )
        membership_predictor.train()

        prediction_vectors = torch.cat(prediction_vectors)
        membership_label_vector = torch.cat(membership_label_vector)
        membership_predictor_dataloader = DataLoader(
            TensorDataset(prediction_vectors, membership_label_vector), batch_size=self.membership_predictor_batch_size
        )
        optimizer = torch.optim.Adam(membership_predictor.parameters(), lr=self.membership_predictor_lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        accuracy = BinaryClassifierAccuracy()

        # train the membership predictor network on the combined dataset by iterating
        # each original image and getting the augmented images of this image back
        early_stopper = EarlyStopper(
            window=self.membership_predictor_early_stopping_window,
            min_diff=self.membership_predictor_early_stopping_mind_diff
        )
        epoch = 0
        last_loss = early_stopper.best_value
        best_acc = 0
        best_model_state_dict = None
        while not early_stopper.stop_early(last_loss):
            print(f'Epoch {epoch + 1}')

            running_loss = 0.0
            accuracy.reset()
            for idx, (batch, member_labels) in enumerate(membership_predictor_dataloader):
                batch, member_labels = batch.to(
                    membership_predictor.device), member_labels.to(membership_predictor.device)
                optimizer.zero_grad()
                outputs = membership_predictor(batch.float())

                loss = criterion(outputs, member_labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch.size(0)
                accuracy.update(outputs.sigmoid(), member_labels)

            training_loss = running_loss / len(membership_predictor_dataloader)
            last_loss = training_loss
            epoch += 1

            if best_acc < accuracy.compute_metric():
                best_model_state_dict = membership_predictor.state_dict()
                best_acc = accuracy.compute_metric()

            print(f'Epoch loss: {training_loss:.4f} \t Training Accuracy: {accuracy.compute_metric():.4f}')

        print(f'Using best model with training accuracy of {best_acc:.4f}')
        membership_predictor.load_state_dict(best_model_state_dict)

        return membership_predictor

    def evaluate(self, target_membership_set, target_non_membership_set):
        print(
            f'Start evaluation on {len(target_membership_set)} membership and {len(target_non_membership_set)} non-membership samples'
        )

        # create the augmented datasets
        augmented_membership_set = AugmentationDataset(target_membership_set, d=self.d, r=self.r)
        augmented_non_membership_set = AugmentationDataset(target_non_membership_set, d=self.d, r=self.r)

        # get the correct/wrong predictions of the target model as a vector of bool values
        target_model_membership_outputs = self.infer(augmented_membership_set, self.target_model)
        target_model_non_membership_outputs = self.infer(augmented_non_membership_set, self.target_model)

        # feed the bool vectors into the membership predictor
        membership_dataloader = DataLoader(
            TensorDataset(target_model_membership_outputs), batch_size=self.membership_predictor_batch_size
        )
        non_membership_dataloader = DataLoader(
            TensorDataset(target_model_non_membership_outputs), batch_size=self.membership_predictor_batch_size
        )

        with torch.no_grad():
            self.membership_predictor.eval()
            correct_memberships = 0
            for batch in tqdm(membership_dataloader, desc="Evaluating on membership data"):
                membership_predictions = self.membership_predictor(batch[0].float()).sigmoid()
                correct_memberships += (membership_predictions >= 0.5).sum().item()

            correct_non_memberships = 0
            for batch in tqdm(non_membership_dataloader, desc="Evaluating on membership data"):
                membership_predictions = self.membership_predictor(batch[0].float()).sigmoid()
                correct_non_memberships += (membership_predictions < 0.5).sum().item()

        # calculate the accuracy
        acc = (correct_memberships + correct_non_memberships) / \
              (len(target_membership_set) + len(target_non_membership_set))
        acc_membership = correct_memberships / len(target_membership_set)
        acc_non_membership = correct_non_memberships / \
                             len(target_non_membership_set)

        print(
            f'Evaluation result: acc={acc:.4f}, acc_membership={acc_membership:.4f}, acc_non_membership={acc_non_membership:.4f}'
        )

        return acc, acc_membership, acc_non_membership

    def infer(self, dataset, target_model, **kwargs):
        if target_model is None:
            target_model = self.target_model
        dataloader = DataLoader(
            dataset, batch_size=self.target_model_inference_batch_size, num_workers=self.data_loader_workers
        )
        prediction_vectors = []
        with torch.no_grad():
            target_model.eval()
            for augmented_images, class_labels, _ in tqdm(dataloader, desc="Inferring"):
                image_tensor, class_labels = augmented_images.to(
                    target_model.device), class_labels.to(target_model.device)
                image_tensor = image_tensor.squeeze().view(
                    -1, augmented_images.size()[-3], augmented_images.size()[-2], augmented_images.size()[-1]
                )
                class_labels = class_labels.unsqueeze(1)
                logits = target_model(image_tensor).view(-1, self.number_of_augmented_images, target_model.num_classes)
                predictions = torch.argmax(logits, dim=2)

                prediction_vectors.append(predictions == class_labels)

                if self.rtpt is not None:
                    self.rtpt.step()

        return torch.cat(prediction_vectors)


class GapAttack(MembershipInferenceAttack):
    def __init__(self, target_model, rtpt: RTPT, mix_validation_and_train_set=False):
        """
        https://arxiv.org/abs/2007.14321
        """
        super().__init__(target_model, 'GapAttack', rtpt, mix_validation_and_train_set)

    def setup_attack(self, target_train_config_helper, dataset_size, batch_size=128):
        self.batch_size = batch_size

        # Set seed
        seed = target_train_config_helper.config.seed
        torch.manual_seed(seed)

        # get the dataset and split in in half
        train_set, valid_set, test_set = target_train_config_helper.get_pytorch_dataset(
            target_train_config_helper.dataset_config.args, seed=seed, shadow_datasets=True)

        self.target_train_set, self.shadow_train_set = train_set
        self.target_valid_set, self.shadow_valid_set = valid_set
        self.target_test_set, self.shadow_test_set = test_set

        self.target_membership_set, self.target_non_membership_set = self.get_membership_splits(
            self.target_train_set,
            self.target_valid_set,
            self.target_test_set,
            dataset_size, seed,
            self.mix_validation_and_train_set
        )
        self.shadow_membership_set, self.shadow_non_membership_set = self.get_membership_splits(
            self.shadow_train_set,
            self.shadow_valid_set,
            self.shadow_test_set,
            dataset_size, seed,
            self.mix_validation_and_train_set
        )

        self.rtpt.max_iterations = len(self.target_membership_set) + len(self.target_non_membership_set)

        return self.target_membership_set, self.target_non_membership_set

    def evaluate(self, membership_data, non_membership_data):
        print(
            f'Start evaluation on {len(membership_data)} membership and {len(non_membership_data)} non-membership samples'
        )
        correct_memberships = np.sum(np.array(self.infer(membership_data)) == True)
        correct_non_memberships = np.sum(np.array(self.infer(non_membership_data)) == False)
        acc = (correct_memberships + correct_non_memberships) / \
              (len(membership_data) + len(non_membership_data))
        acc_membership = correct_memberships / len(membership_data)
        acc_non_membership = correct_non_memberships / len(non_membership_data)
        print(
            f'Evaluation result: acc={acc:.4f}, acc_membership={acc_membership:.4f}, acc_non_membership={acc_non_membership:.4f}'
        )
        return acc, acc_membership, acc_non_membership

    def infer(self, dataset, target_model=None):
        """
        Predicts for samples X if they were part of the training set of the target model.
        Returns True if membership is predicted, False else.
        """
        if target_model is None:
            target_model = self.target_model
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=2)
        predictions = []
        target_model.eval()
        for X, y in dataloader:
            X, y = X.to(target_model.device), y.to(target_model.device)
            with torch.no_grad():
                y_pred = torch.argmax(target_model.forward(X), dim=1)
                predictions.append(y_pred == y)

            if self.rtpt is not None:
                self.rtpt.step()
        return torch.cat(predictions).cpu().tolist()


class DecisionBoundaryAttack(MembershipInferenceAttack):
    def __init__(self, target_model, rtpt: RTPT, mix_validation_and_train_set=False):
        """
        Create a `LabelOnlyDecisionBoundary` instance for Label-Only Inference Attack based on Decision Boundary.
        :param estimator: A trained classification estimator.
        :param distance_threshold_tau: Threshold distance for decision boundary. Samples with boundary distances larger
                                       than threshold are considered members of the training dataset.
        """
        super().__init__(target_model, 'DecisionBoundaryAttack', rtpt, mix_validation_and_train_set)

    def setup_attack(
        self,
        train_config_helper: TrainConfigHelper,
        dataset_size: int,
        use_shadow_model=True,
        shadow_model_run_path: str = "",
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        tau: float = 0.5,
        batch_size: int = 128,
        max_iter: int = 10,
        max_eval: int = 500,
        init_eval: int = 10,
        init_size: int = 100,
        relabel_shadow_dataset=False,
        early_stopping_relabelled_shadow_training={
            "window": 5, "min_diff": 0.005
        },
        retrain_relabelled_shadow_model=True
    ) -> Tuple[Dataset, Dataset]:
        self.batch_size = batch_size
        self.tau = tau
        self.input_shape = input_shape
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.init_size = 100
        self.relabel_shadow_dataset = relabel_shadow_dataset
        self.early_stopping_relabelled_shadow_training = early_stopping_relabelled_shadow_training

        # Set seed
        seed = train_config_helper.config.seed
        torch.manual_seed(seed)

        # get the dataset and split in in half
        train_set, valid_set, test_set = train_config_helper.get_pytorch_dataset(
            train_config_helper.dataset_config.args, seed=seed, shadow_datasets=True)

        self.target_train_set, self.shadow_train_set = train_set
        self.target_valid_set, self.shadow_valid_set = valid_set
        self.target_test_set, self.shadow_test_set = test_set

        self.attacked_model = self.target_model
        if use_shadow_model:
            self.attacked_model = get_shadow_model(
                self.shadow_train_set,
                self.shadow_valid_set,
                self.target_model,
                shadow_model_run_path,
                relabel_shadow_dataset,
                early_stopping_relabelled_shadow_training,
                retrain_relabelled_shadow_model,
                self.batch_size,
                self.rtpt
            )

        self.target_membership_set, self.target_non_membership_set = self.get_membership_splits(
            self.target_train_set,
            self.target_valid_set,
            self.target_test_set,
            dataset_size, seed,
            self.mix_validation_and_train_set
        )
        self.shadow_membership_set, self.shadow_non_membership_set = self.get_membership_splits(
            self.shadow_train_set,
            self.shadow_valid_set,
            self.shadow_test_set,
            dataset_size, seed,
            self.mix_validation_and_train_set
        )

        self.estimate_parameters(
            membership_data=self.shadow_membership_set, non_membership_data=self.shadow_non_membership_set
        )

        self.rtpt.max_iterations = len(self.target_membership_set) + len(self.target_non_membership_set) + \
                                   len(self.shadow_membership_set) + len(self.shadow_non_membership_set)

        return self.target_membership_set, self.target_non_membership_set

    def infer(self, dataset, target_model=None, **kwargs):
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
        x = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset))))[0].numpy()
        y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset))))[1].numpy()

        if target_model is None:
            target_model = self.target_model

        hsj = HopSkipJump(classifier=target_model, input_shape=self.input_shape, rtpt=self.rtpt, **kwargs)

        x_adv = hsj.generate(x=x, y=y)
        distance = np.linalg.norm((x_adv - x).reshape((x.shape[0], -1)), ord=2, axis=1)

        y_pred = self.target_model.predict(x, numpy=True)

        distance[y_pred != y] = 0

        is_member = np.where(distance > self.tau, 1, 0)

        return torch.tensor(is_member)

    def estimate_parameters(self, membership_data, non_membership_data):
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
        x_train = next(iter(torch.utils.data.DataLoader(membership_data, batch_size=len(membership_data))))[0].numpy()
        y_train = next(iter(torch.utils.data.DataLoader(membership_data, batch_size=len(membership_data))))[1].numpy()
        x_test = next(iter(torch.utils.data.DataLoader(non_membership_data,
                                                       batch_size=len(non_membership_data))))[0].numpy()
        y_test = next(iter(torch.utils.data.DataLoader(non_membership_data,
                                                       batch_size=len(non_membership_data))))[1].numpy()

        hsj = HopSkipJump(classifier=self.attacked_model, input_shape=self.input_shape, rtpt=self.rtpt)

        x_train_adv = hsj.generate(x=x_train, y=y_train)
        x_test_adv = hsj.generate(x=x_test, y=y_test)

        distance_train = np.linalg.norm((x_train_adv - x_train).reshape((x_train.shape[0], -1)), ord=2, axis=1)
        distance_test = np.linalg.norm((x_test_adv - x_test).reshape((x_test.shape[0], -1)), ord=2, axis=1)

        y_train_pred = self.attacked_model.predict(x_train, numpy=True)
        y_test_pred = self.attacked_model.predict(x_test, numpy=True)

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

        print(f'Setting threshold to {distance_threshold_tau}')
        self.tau = distance_threshold_tau

    def evaluate(self, membership_data, non_membership_data):
        print(
            f'Start evaluation on {len(membership_data)} membership and {len(non_membership_data)} non-membership samples using threshold {self.tau}'
        )
        correct_memberships = np.sum(np.array(self.infer(membership_data)) == True)
        correct_non_memberships = np.sum(np.array(self.infer(non_membership_data)) == False)
        acc = (correct_memberships + correct_non_memberships) / \
              (len(membership_data) + len(non_membership_data))
        acc_membership = correct_memberships / len(membership_data)
        acc_non_membership = correct_non_memberships / len(non_membership_data)
        print(
            f'Evaluation result: acc={acc:.4f}, acc_membership={acc_membership:.4f}, acc_non_membership={acc_non_membership:.4f}'
        )
        return acc, acc_membership, acc_non_membership

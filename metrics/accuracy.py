import torch
from metrics.base_metric import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, name='accuracy'):
        super().__init__(name)

    def compute_metric(self):
        accuracy = self._num_corrects / self._num_samples
        return accuracy


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


class MixUpAccuracy(BaseMetric):
    def __init__(self, name: str = 'mixup_accuracy'):
        super().__init__(name)

    def compute_metric(self):
        return self._num_corrects / self._num_samples

    def update(self, model_output: torch.tensor, y_true_a: torch.tensor, y_true_b: torch.tensor, lmbda: float):
        predictions = torch.argmax(model_output, dim=1)
        self._num_corrects += lmbda * torch.eq(predictions, y_true_a).sum().item()
        self._num_corrects += (1 - lmbda) * torch.eq(predictions, y_true_b).sum().item()
        self._num_samples += model_output.shape[0]

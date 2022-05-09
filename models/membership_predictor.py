from models.base_model import BaseModel
import torch.nn as nn


class MembershipPredictor(BaseModel):
    """
    Simple Neural Network which gets the info for each augmented datapoint whether the model misclassified the augmented images or not
    and outputs whether the image should be classified as a member of the training set or not.
    """
    def __init__(
        self, input_dim, hidden_layer_list, name="MembershipPredictor", activation_function=nn.ReLU, *args, **kwargs
    ):
        super().__init__(name, *args, **kwargs)
        self.input_dim = input_dim

        # create the NN according to the paper "Label-Only Membership Inference Attacks"
        modules = [nn.Linear(self.input_dim, hidden_layer_list[0]), activation_function()]
        for num_neurons in hidden_layer_list[1:]:
            modules.append(nn.Linear(modules[-2].out_features, num_neurons))
            modules.append(activation_function())
        modules.append(nn.Linear(modules[-2].out_features, 1))
        self.model = nn.Sequential(*modules)

        self.to(self.device)

    def forward(self, X):
        return self.model(X)

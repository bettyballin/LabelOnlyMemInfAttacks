from abc import abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from .attack import PredictionScoreAttack
from utils.training import EarlyStopper


class BoundaryAttack(PredictionScoreAttack):
    def __init__(
        self, 
        apply_softmax: bool,
        batch_size: int = 128,
        log_training: bool = False
        ):
        super().__init__('BoundaryAttack')

        self.apply_softmax = apply_softmax
        self.batch_size = batch_size
        self.log_training = log_training


    def learn_attack_parameters(self, shadow_model: nn.Module, member_dataset: Dataset, non_member_dataset: Dataset):
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

        return torch.from_numpy(predictions*1)

def prediction(x):
    x_list = x[0].tolist()
    x_sort = sorted(x_list)
    max_index = x_list.index(x_sort[-1])

    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum

    return softmax, max_index

def AdversaryOne_Feature(args, shadowmodel, data_loader, cluster, Statistic_Data):
    Loss = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.cuda()
            output = shadowmodel(data)
            Loss.append(F.cross_entropy(output, target.cuda()).item())
    Loss = np.asarray(Loss)
    half = int(len(Loss)/2)
    member = Loss[:half]
    non_member = Loss[half:]        
    for loss in member:
        Statistic_Data.append({'DataSize':float(cluster), 'Loss':loss,  'Status':'Member'})
    for loss in non_member:
        Statistic_Data.append({'DataSize':float(cluster), 'Loss':loss,  'Status':'Non-member'})
    return Statistic_Data


def AdversaryOne_evaluation(args, targetmodel, shadowmodel, data_loader, cluster, AUC_Loss, AUC_Entropy, AUC_Maximum):
    Loss = []
    Entropy = []
    Maximum = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            Toutput = targetmodel(data)
            Tlabel = Toutput.max(1)[1]

            Soutput = shadowmodel(data)
            if Tlabel != target:
               
                Loss.append(100)
            else:
                Loss.append(F.cross_entropy(Soutput, target).item())
            
            prob = F.softmax(Soutput, dim=1) 

            Maximum.append(torch.max(prob).item())
            entropy = -1 * torch.sum(torch.mul(prob, torch.log(prob)))
            if str(entropy.item()) == 'nan':
                Entropy.append(1e-100)
            else:
                Entropy.append(entropy.item())
 
    mem_groundtruth = np.ones(int(len(data_loader.dataset)/2))
    non_groundtruth = np.zeros(int(len(data_loader.dataset)/2))
    groundtruth = np.concatenate((mem_groundtruth, non_groundtruth))

    predictions_Loss = np.asarray(Loss)
    predictions_Entropy = np.asarray(Entropy)
    predictions_Maximum = np.asarray(Maximum)
    
    
def AdversaryTwo_HopSkipJump(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data=False, maxitr=50, max_eval=10000):
    input_shape = [(3, 32, 32), (3, 32, 32), (3, 64, 64), (3, 128, 128)]
    nb_classes = [10, 100, 43, 19]
    ARTclassifier = PyTorchClassifier(
                model=targetmodel,
                clip_values=(0, 1),
                loss=F.cross_entropy,
                input_shape=input_shape[args.dataset_ID],
                nb_classes=nb_classes[args.dataset_ID],
            )
    L0_dist, L1_dist, L2_dist, Linf_dist = [], [], [], []
    Attack = HopSkipJump(classifier=ARTclassifier, targeted =False, max_iter=maxitr, max_eval=max_eval)

    mid = int(len(data_loader.dataset)/2)
    member_groundtruth, non_member_groundtruth = [], []
    for idx, (data, target) in enumerate(data_loader): 
        targetmodel.module.query_num = 0
        data = np.array(data)  
        logit = ARTclassifier.predict(data)
        _, pred = prediction(logit)
        if pred != target.item() and not Random_Data:
            success = 1
            data_adv = data
        else:
            data_adv = Attack.generate(x=data) 
            data_adv = np.array(data_adv) 
            if Random_Data:
                success = compute_success(ARTclassifier, data, [pred], data_adv) 
            else:
                success = compute_success(ARTclassifier, data, [target.item()], data_adv)

        if success == 1:
            print(targetmodel.module.query_num)
            logx.msg('-------------Training DataSize: {} current img index:{}---------------'.format(cluster, idx))
            L0_dist.append(l0(data, data_adv))
            L1_dist.append(l1(data, data_adv))
            L2_dist.append(l2(data, data_adv))
            Linf_dist.append(linf(data, data_adv))

            if idx < mid:
                member_groundtruth.append(1)
            else:
                non_member_groundtruth.append(0)

        if Random_Data and len(L0_dist)==100:
            break
        
def AdversaryTwo_QEBA(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data=False, max_iter=150):
    #input_shape = [(3, 32, 32), (3, 32, 32), (3, 64, 64), (3, 128, 128), (3, 64, 64)]
    nb_classes = [10, 100, 43, 19, 200]
    PGEN = ['resize768']
    p_gen, maxN, initN = load_pgen(args, PGEN[0])
    
    fmodel = QEBA.models.PyTorchModel(targetmodel, bounds=(0, 1), 
                num_classes=nb_classes[args.dataset_ID], discretize=False)
    Attack = QEBA.attacks.BAPP_custom(fmodel, criterion=Misclassification()) #criterion=TargetClass(src_label)
    L0_dist, L1_dist, L2_dist, Linf_dist = [], [], [], []
    member_groundtruth, non_member_groundtruth = [], []
    mid = int(len(data_loader.dataset)/2)
    for idx, (data, target) in enumerate(data_loader):   
        targetmodel.module.query_num = 0
        data = data.numpy()
        data = np.squeeze(data)
        pred = np.argmax(fmodel.forward_one(data))
        if pred != target.item():
            data_adv = data
            pred_adv = pred
        else:
            grad_gt = fmodel.gradient_one(data, label=target.item())
            rho = p_gen.calc_rho(grad_gt, data).item()

            Adversarial = Attack(data, label=target.item(), starting_point = None, iterations=max_iter, stepsize_search='geometric_progression', 
                        unpack=False, max_num_evals=maxN, initial_num_evals=initN, internal_dtype=np.float32, 
                        rv_generator = p_gen, atk_level=999, mask=None, batch_size=1, rho_ref = rho, 
                        log_every_n_steps=1, suffix=PGEN[0], verbose=False)  

        
            data_adv = Adversarial.perturbed     
            pred_adv = Adversarial.adversarial_class

        if target.item() != pred_adv and type(data_adv) == np.ndarray:
            print(targetmodel.module.query_num)
            logx.msg('-------------Training DataSize: {} current img index:{}---------------'.format(cluster, idx))
            data = data[np.newaxis, :]
            data_adv = data_adv[np.newaxis, :]
            L0_dist.append(l0(data, data_adv))
            L1_dist.append(l1(data, data_adv))
            L2_dist.append(l2(data, data_adv))
            Linf_dist.append(linf(data, data_adv))
            if idx < mid:
                member_groundtruth.append(1)
            else:
                non_member_groundtruth.append(0)
        if Random_Data and len(L0_dist)==100:
            break
    member_groundtruth = np.array(member_groundtruth)
    non_member_groundtruth = np.array(non_member_groundtruth)

    groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))
    L0_dist = np.asarray(L0_dist)
    L1_dist = np.asarray(L1_dist)
    L2_dist = np.asarray(L2_dist)
    Linf_dist = np.asarray(Linf_dist)

    fpr, tpr, _ = roc_curve(groundtruth, L0_dist, pos_label=1, drop_intermediate=False)
    L0_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L1_dist, pos_label=1, drop_intermediate=False)
    L1_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L2_dist, pos_label=1, drop_intermediate=False)
    L2_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, Linf_dist, pos_label=1, drop_intermediate=False)
    Linf_auc = round(auc(fpr, tpr), 4)

    ### AUC based on distance
    auc_score = {'DataSize':float(cluster), 'L0_auc':L0_auc, 'L1_auc':L1_auc, 'L2_auc':L2_auc, 'Linf_auc':Linf_auc}
    AUC_Dist.append(auc_score)

    ### Distance of L0, L1, L2, Linf
    middle= int(len(L0_dist)/2)
    for idx, (l0_dist, l1_dist, l2_dist, linf_dist) in enumerate(zip(L0_dist, L1_dist, L2_dist, Linf_dist)):   
        if idx < middle:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Member'}
        else:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Non-member'}
        Distance.append(data)
    return AUC_Dist, Distance
    
def AdversaryTwo_SaltandPepperNoise(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data=False, max_iter=150):
    nb_classes = [10, 100, 43, 19, 200]
    PGEN = ['resize768']
    # p_gen, maxN, initN = load_pgen(args, PGEN[0])
    
    fmodel = QEBA.models.PyTorchModel(targetmodel, bounds=(0, 1), 
                num_classes=nb_classes[args.dataset_ID], discretize=False)
    Attack = QEBA.attacks.SaltAndPepperNoiseAttack(fmodel, criterion=Misclassification()) #criterion=TargetClass(src_label)
    L0_dist, L1_dist, L2_dist, Linf_dist = [], [], [], []
    member_groundtruth, non_member_groundtruth = [], []
    mid = int(len(data_loader.dataset)/2)
    for idx, (data, target) in enumerate(data_loader):   
        targetmodel.module.query_num = 0
        data = data.numpy()
        data = np.squeeze(data)
        pred = np.argmax(fmodel.forward_one(data))
   
        if pred != target.item():
            data_adv = data
            pred_adv = pred
        else:

            data_adv = Attack(data, label=target.item())  

            if type(data_adv) == np.ndarray:
                pred_adv = np.argmax(fmodel.forward_one(data_adv))
            else:
                continue
        if target.item() != pred_adv:
            print(targetmodel.module.query_num)
            logx.msg('-------------Training DataSize: {} current img index:{}---------------'.format(cluster, idx))
            data = data[np.newaxis, :]
            data_adv = data_adv[np.newaxis, :]
            L0_dist.append(l0(data, data_adv))
            L1_dist.append(l1(data, data_adv))
            L2_dist.append(l2(data, data_adv))
            Linf_dist.append(linf(data, data_adv))
            if idx < mid:
                member_groundtruth.append(1)
            else:
                non_member_groundtruth.append(0)
        if Random_Data and len(L0_dist)==100:
            break
    member_groundtruth = np.array(member_groundtruth)
    non_member_groundtruth = np.array(non_member_groundtruth)

    groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))
    L0_dist = np.asarray(L0_dist)
    L1_dist = np.asarray(L1_dist)
    L2_dist = np.asarray(L2_dist)
    Linf_dist = np.asarray(Linf_dist)

    fpr, tpr, _ = roc_curve(groundtruth, L0_dist, pos_label=1, drop_intermediate=False)
    L0_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L1_dist, pos_label=1, drop_intermediate=False)
    L1_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L2_dist, pos_label=1, drop_intermediate=False)
    L2_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, Linf_dist, pos_label=1, drop_intermediate=False)
    Linf_auc = round(auc(fpr, tpr), 4)

    ### AUC based on distance
    auc_score = {'DataSize':float(cluster), 'L0_auc':L0_auc, 'L1_auc':L1_auc, 'L2_auc':L2_auc, 'Linf_auc':Linf_auc}
    AUC_Dist.append(auc_score)

    ### Distance of L0, L1, L2, Linf
    middle= int(len(L0_dist)/2)
    for idx, (l0_dist, l1_dist, l2_dist, linf_dist) in enumerate(zip(L0_dist, L1_dist, L2_dist, Linf_dist)):   
        if idx < middle:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Member'}
        else:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Non-member'}
        Distance.append(data)
    return AUC_Dist, Distance
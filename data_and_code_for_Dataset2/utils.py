from torch import nn
from param import parameter_parser
args = parameter_parser()
import torch


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
    def forward(self, one_index, zero_index, input, target):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1-args.beta)*loss_sum[one_index].sum() + args.beta*loss_sum[zero_index].sum()


def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg


class kl_loss(nn.Module):
    def __init__(self, num_diseases, num_lncRNAs):
        super(kl_loss, self).__init__()
        self.num_diseases = num_diseases
        self.num_lncRNAs = num_lncRNAs
    def forward(self, z_disease_log_std, z_disease_mean, z_lncRNA_log_std, z_lncRNA_mean):
        kl_disease = - (0.5 / self.num_diseases) * torch.mean(torch.sum(
            1 + 2 * z_disease_log_std - torch.pow(z_disease_mean, 2) - torch.pow(torch.exp(z_disease_log_std), 2),
            1))
        kl_lncRNA = - (0.5 / self.num_lncRNAs) * torch.mean(
            torch.sum(
                1 + 2 * z_lncRNA_log_std - torch.pow(z_lncRNA_mean, 2) - torch.pow(torch.exp(z_lncRNA_log_std), 2), 1))
        kl = kl_disease + kl_lncRNA
        return kl




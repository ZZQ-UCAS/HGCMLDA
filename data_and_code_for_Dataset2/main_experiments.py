import torch
from prepareData import prepare_data
import numpy as np
from torch import optim
from param import parameter_parser
from Module import HGCMLDA
from utils import get_L2reg, Myloss,kl_loss
from Calculate_Metrics import Metric_fun
from trainData import Dataset
import ConstructHW
from scipy import sparse as sp
import torch.nn.functional as F


import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_epoch(model, train_data, optim, opt):
    model.train()
    regression_crit = Myloss()
    one_index = train_data[2][0].to(device).t().tolist()
    zero_index = train_data[2][1].to(device).t().tolist()
    dis_sim_integrate_tensor = train_data[0].to(device)
    lnc_sim_integrate_tensor = train_data[1].to(device)
    concat_lncRNA = lnc_sim_integrate_tensor.detach().cpu().numpy()
    concat_lnc_tensor = torch.FloatTensor(concat_lncRNA)
    concat_lnc_tensor = concat_lnc_tensor.to(device)
    G_lnc_Kn = ConstructHW.constructHW_knn(concat_lnc_tensor.detach().cpu().numpy(), K_neigs=[14], is_probH=False)
    G_lnc_Gm = ConstructHW.constructHW_gmm(concat_lnc_tensor.detach().cpu().numpy(), clusters=[9])
    G_lnc_Kn = G_lnc_Kn.to(device)
    G_lnc_Gm = G_lnc_Gm.to(device)
    concat_dis = dis_sim_integrate_tensor.detach().cpu().numpy()
    concat_dis_tensor = torch.FloatTensor(concat_dis)
    concat_dis_tensor = concat_dis_tensor.to(device)
    G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.detach().cpu().numpy(), K_neigs=[14], is_probH=False)
    G_dis_Gm = ConstructHW.constructHW_gmm(concat_dis_tensor.detach().cpu().numpy(), clusters=[9])
    G_dis_Kn = G_dis_Kn.to(device)
    G_dis_Gm = G_dis_Gm.to(device)
    loss_kl = kl_loss(310, 2853)
    for epoch in range(1, opt.epoch+1):
        score, lnc_cl_loss, dis_cl_loss,reconstruction_VAE = model(concat_lnc_tensor, concat_dis_tensor,
                                                G_lnc_Kn, G_lnc_Gm, G_dis_Kn, G_dis_Gm,train_data[7].to(device),train_data[7].T.to(device))
        loss_k = loss_kl(model.z_disease_log_std, model.z_disease_mean, model.z_lncRNA_log_std, model.z_lncRNA_mean)
        reconstruction_VAE=reconstruction_VAE.float()
        loss_v = loss_k + regression_crit(one_index, zero_index, train_data[7].to(device),reconstruction_VAE)
        recover_loss = regression_crit(one_index, zero_index, train_data[7].to(device), score)
        reg_loss = get_L2reg(model.parameters())
        tol_loss = recover_loss + 0.1*(lnc_cl_loss + dis_cl_loss)+0.15*loss_v
        optim.zero_grad()
        tol_loss.backward()
        optim.step()
    true_value_one, true_value_zero, pre_value_one, pre_value_zero,true_value_one_independent, true_value_zero_independent, pre_value_one_independent, pre_value_zero_independent = test(model, train_data, concat_lnc_tensor, concat_dis_tensor,
                                 G_lnc_Kn, G_lnc_Gm, G_dis_Kn, G_dis_Gm)
    return true_value_one, true_value_zero, pre_value_one, pre_value_zero,true_value_one_independent, true_value_zero_independent, pre_value_one_independent, pre_value_zero_independent

def test(model, data, concat_lnc_tensor, concat_dis_tensor, G_lnc_Kn, G_lnc_Gm, G_dis_Kn, G_dis_Gm):
    model.eval()
    score,_,_,_ = model(concat_lnc_tensor, concat_dis_tensor,G_lnc_Kn, G_lnc_Gm, G_dis_Kn, G_dis_Gm,data[7].to(device),data[7].T.to(device))    
    test_one_index = data[3][0].t().tolist()
    test_zero_index = data[3][1].t().tolist()
    true_one = data[5][test_one_index]
    true_zero = data[5][test_zero_index]
    pre_one = score[test_one_index]
    pre_zero = score[test_zero_index]
    test_one_index_independent = data[6][0].t().tolist()
    test_zero_index_independent = data[6][1].t().tolist()
    true_one_independent = data[5][test_one_index_independent]
    true_zero_independent = data[5][test_zero_index_independent]
    pre_one_independent = score[test_one_index_independent]
    pre_zero_independent = score[test_zero_index_independent]
    return true_one, true_zero, pre_one, pre_zero,true_one_independent,true_zero_independent,pre_one_independent,pre_zero_independent


def evaluate_validation(true_one, true_zero, pre_one, pre_zero,flag):
    Metric = Metric_fun()
    seed=4
    test_po_num = true_one.shape[0]
    test_index = np.array(np.where(true_zero == 0))
    np.random.seed(seed)
    np.random.shuffle(test_index.T)
    test_ne_index = tuple(test_index[:, :test_po_num])
    eval_true_zero = true_zero[test_ne_index]
    eval_true_data = torch.cat([true_one,eval_true_zero])
    eval_pre_zero = pre_zero[test_ne_index]
    eval_pre_data = torch.cat([pre_one,eval_pre_zero])
    eval_true_data_save=eval_true_data.detach().cpu().numpy().flatten()
    eval_pre_data_save=eval_pre_data.detach().cpu().numpy().flatten()
    np.savetxt(str(flag+1)+'true_valid.txt',eval_true_data_save,fmt='%f',delimiter=' ')
    np.savetxt(str(flag+1)+'predict_valid.txt',eval_pre_data_save,fmt='%f',delimiter=' ')
    print(Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data))
    metrics_tensor = Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)
    return metrics_tensor



def evaluate_test(true_one_independent, true_zero_independent, pre_one_independent, pre_zero_independent,flag):
    Metric2 = Metric_fun()
    test_po_num_independent = true_one_independent.shape[0]
    test_index_independent = np.array(np.where(true_zero_independent == 0))
    np.random.shuffle(test_index_independent.T)
    test_ne_index_independent = tuple(test_index_independent[:, :test_po_num_independent])
    eval_true_zero2 = true_zero_independent[test_ne_index_independent]
    eval_true_data2 = torch.cat([true_one_independent,eval_true_zero2])
    eval_pre_zero2 = pre_zero_independent[test_ne_index_independent]
    eval_pre_data2 = torch.cat([pre_one_independent,eval_pre_zero2])
    eval_true_data2_save=eval_true_data2.detach().cpu().numpy().flatten()
    eval_pre_data2_save=eval_pre_data2.detach().cpu().numpy().flatten()
    np.savetxt(str(flag+1)+'true_ind_test.txt',eval_true_data2_save,fmt='%f',delimiter=' ')
    np.savetxt(str(flag+1)+'predict_ind_test.txt',eval_pre_data2_save,fmt='%f',delimiter=' ')
    metrics_tensor_independent = Metric2.cv_mat_model_evaluate(eval_true_data2, eval_pre_data2)
    print(metrics_tensor_independent)
    return metrics_tensor_independent


def main(opt):
    dataset = prepare_data(opt)
    train_data = Dataset(opt, dataset)
    metrics_cross = np.zeros((1, 6))
    metrics_independent=np.zeros((1,6))
    for i in range(opt.validation):
        hidden_list = [128, 128]
        num_proj_hidden = 64
        model = HGCMLDA(args.lnc_num, args.dis_num, hidden_list, num_proj_hidden, args)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr = 0.0001)
        true_score_one, true_score_zero, pre_score_one, pre_score_zero,true_score_one_independent, true_score_zero_independent, pre_score_one_independent, pre_score_zero_independent = train_epoch(model, train_data[i], optimizer,opt)
        metrics_value = evaluate_validation(true_score_one, true_score_zero, pre_score_one, pre_score_zero,i)
        metrics_cross = metrics_cross + metrics_value
        metrics_value2=evaluate_test(true_score_one_independent, true_score_zero_independent, pre_score_one_independent, pre_score_zero_independent,i)
        metrics_independent=metrics_independent+metrics_value2
    metrics_cross_avg = metrics_cross / opt.validation
    metrics_independent_avg=metrics_independent/5
    print('metrics_avg_CV:',metrics_cross_avg)
    print('metrics_avg_ind_test:',metrics_independent_avg)



if __name__ == '__main__':
    args = parameter_parser()
    main(args)


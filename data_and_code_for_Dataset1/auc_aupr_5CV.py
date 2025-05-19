import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve,precision_recall_curve,auc
from numpy import interp
import warnings



tprs = []
mean_fpr = np.linspace(0, 1, 1000)


y_real = []
y_proba = [] 


for temp in range(5):
    y_true=np.loadtxt(str(temp+1)+'true_valid.txt',dtype=np.int32)
    y_pred=np.loadtxt(str(temp+1)+'predict_valid.txt',dtype=np.float64)

    fpr, tpr, thresholds= roc_curve(y_true, y_pred)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

    precision, recall, thresholds2 = precision_recall_curve(y_true, y_pred)
    y_real.append(y_true)
    y_proba.append(y_pred)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0

#save_fpr=np.array(mean_fpr)
#np.savetxt('HGCMLDA_fpr_5CV.txt',save_fpr,fmt='%f',delimiter=' ')
#save_tpr=np.array(mean_tpr)
#np.savetxt('HGCMLDA_tpr_5CV.txt',save_tpr,fmt='%f',delimiter=' ')

mean_auc = auc(mean_fpr, mean_tpr)
print('auc_5-CV:')
print(mean_auc)

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
    
print('aupr_5-CV:')
print(auc(recall,precision))

#save_precision=np.array(precision)
#np.savetxt('HGCMLDA_precision_5CV.txt',save_precision,fmt='%f',delimiter=' ')
#save_recall=np.array(recall)
#np.savetxt('HGCMLDA_recall_5CV.txt',save_recall,fmt='%f',delimiter=' ')


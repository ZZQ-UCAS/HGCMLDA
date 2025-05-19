import numpy as np

class Metric_fun(object):
    def __init__(self):
        super(Metric_fun).__init__()


    def cv_mat_model_evaluate(self, association_mat, predict_mat):
        
        real_score = np.mat(association_mat.detach().cpu().numpy().flatten())
        predict_score = np.mat(predict_mat.detach().cpu().numpy().flatten())
        return self.get_metrics(real_score, predict_score)

    def get_metrics(self, real_score, predict_score):
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))

        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = sorted_predict_score[
            (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]

        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0
        predict_score_matrix[positive_index] = 1

        TP = (predict_score_matrix * real_score.T).A1
        FP = predict_score_matrix.sum(axis=1).A1 - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN

        recall_list = TP / (TP + FN)
        precision_list = TP / (TP + FP)
        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)
        mcc_list = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index]
        accuracy = accuracy_list[max_index]
        specificity = specificity_list[max_index]
        recall = recall_list[max_index]
        precision = precision_list[max_index]
        mcc = mcc_list[max_index]
        return [f1_score, accuracy, recall, specificity, precision,mcc]
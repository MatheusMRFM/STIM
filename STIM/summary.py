import numpy as np
import tensorflow as tf
import constants
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, precision_recall_fscore_support

MAX_TH_VALUES = 100

class Summary():
    def __init__(self, name, writer, interval_score, interval_loss, th_file):
        self.score = []
        self.Q_loss = []
        self.sup_loss = []
        self.name = name
        self.interval_score = interval_score
        self.interval_loss = interval_loss
        self.last_write_score = 0
        self.last_write_loss = 0
        self.writer = writer
        self.y_true = None
        self.y_pred = None
        self.th_file = th_file
        self.threashold = None

    #-------------------------------------------------------------------
    def add_score(self, score):
        self.score.append(score)

    #-------------------------------------------------------------------
    def add_loss(self, q_loss, sup_loss, y_true, y_pred):
        self.Q_loss.append(q_loss)
        self.sup_loss.append(sup_loss)
        if constants.MODEL_TO_TRAIN != constants.RL_MODEL:
            self._update_pred_vec(y_true, y_pred)

    #-------------------------------------------------------------------
    def _get_single_class_pred(self, y, y_pred, class_index):
        col_true = []
        col_pred = []
        for i in range(y.shape[0]):
            col_true.append(y[i, class_index])
            col_pred.append(y_pred[i, class_index])
        return col_true, col_pred

    #-------------------------------------------------------------------
    def _use_threashold(self, y_pred, th):
        for i in range(len(y_pred)):
            if y_pred[i] < th:
                y_pred[i] = 0.0
            else:
                y_pred[i] = 1.0
        return y_pred

    #-------------------------------------------------------------------
    def _update_threashold(self, y,  y_pred, mode, class_index, n_class):
        if mode == constants.TRAIN:
            fpr, tpr, th = roc_curve(y, y_pred)
            target = tpr - fpr
            index = np.argmax(target)
            best_th = th[index]
            
            if self.threashold is None:
                self.threashold = []
                for _ in range(n_class):
                    self.threashold.append([])
                    
            self.threashold[class_index].append(best_th)
            if len(self.threashold[class_index]) > MAX_TH_VALUES:
                self.threashold[class_index] = self.threashold[class_index][1:]

    #-------------------------------------------------------------------
    def _precision_recall_f1_roc(self, y, y_pred, mode):
        precision = np.zeros(y.shape[1])
        recall = np.zeros(y.shape[1])
        f1_score = np.zeros(y.shape[1])
        roc_auc = np.zeros(y.shape[1])
        th = np.zeros(y.shape[1])
        for col in range(y.shape[1]):
            col_true, col_pred = self._get_single_class_pred(y, y_pred, col)
            try:
                roc_auc[col] = roc_auc_score(col_true, col_pred)
                self._update_threashold(col_true, col_pred, mode, col, y.shape[1]) 
                th[col] = self.get_threashold(col)
                col_pred_th = self._use_threashold(col_pred, th[col])
                p, r, f1, s = precision_recall_fscore_support(col_true, col_pred_th)
                precision[col] = np.mean(p)
                recall[col] = np.mean(r)
                f1_score[col] = np.mean(f1)
            except:
                roc_auc[col] = None
                
        return th, precision, recall, f1_score, roc_auc
    
    #-------------------------------------------------------------------
    def get_threashold(self, class_index):
        mean_th = np.mean(self.threashold[class_index])
        return mean_th

    #-------------------------------------------------------------------
    def _update_pred_vec(self, batch_true, batch_pred):
        if self.y_true is None:
            self.y_true = np.array(batch_true)
            self.y_pred = np.array(batch_pred)
        else:
            self.y_true = np.concatenate((self.y_true, np.array(batch_true)), axis=0)
            self.y_pred = np.concatenate((self.y_pred, np.array(batch_pred)), axis=0)
        print(self.y_true.shape)
        print(self.y_pred.shape)

    #-------------------------------------------------------------------
    def reset_score(self):
        self.score = [] 

    #-------------------------------------------------------------------
    def reset_loss(self):
        self.Q_loss = [] 
        self.y_true = None
        self.y_pred = None

    #-------------------------------------------------------------------
    def write_score(self, count):
        if count - self.last_write_score >= self.interval_score:
            self.last_write_score = count
            mean_score = np.mean(self.score)
            summary = tf.Summary()
            name = self.name + '/Train_'
            summary.value.add(tag=name+'Score', simple_value=float(mean_score))

            self.writer.add_summary(summary, count)
            self.writer.flush()
            self.reset_score()

    #-------------------------------------------------------------------
    def write_loss(self, count):
        if count - self.last_write_loss >= self.interval_loss:
            self.last_write_loss = count
            mean_loss_rl = np.mean(self.Q_loss)
            mean_loss_sup = np.mean(self.sup_loss)
            summary = tf.Summary()
            name = self.name + '/Train_'
            summary.value.add(tag=name+'Loss_RL', simple_value=float(mean_loss_rl))
            summary.value.add(tag=name+'Loss_SUP', simple_value=float(mean_loss_sup))

            if constants.MODEL_TO_TRAIN != constants.RL_MODEL:
                th, precision, recall, f1, roc = self._precision_recall_f1_roc(self.y_true, self.y_pred, constants.TRAIN)
                for i in range(len(roc)):
                    if roc[i] != None:
                        print('ROC = ', roc)
                        print('PRECISION = ', precision)
                        summary.value.add(tag=name+'_AUC_ROC_'+str(i), simple_value=float(roc[i]))
                        summary.value.add(tag=name+'_Threashold'+str(i), simple_value=float(th[i]))
                        summary.value.add(tag=name+'_Precision'+str(i), simple_value=float(precision[i]))
                        summary.value.add(tag=name+'_Recall'+str(i), simple_value=float(recall[i]))
                        summary.value.add(tag=name+'_F1_Score'+str(i), simple_value=float(f1[i]))

            self.writer.add_summary(summary, count)
            self.writer.flush()
            self.reset_loss()

    #-------------------------------------------------------------------
    def save_threashold_file(self):
        if not self.threashold is None:
            #df = pd.DataFrame()
            th_vec = []
            for i in range(len(self.threashold)):
                best_th = self.get_threashold(i)
                th_vec.append(best_th)
            th_file = open(self.th_file, 'w')
            for i in range(len(th_vec)):
                th_file.write(str(th_vec[i]) + '\n')
            th_file.close()
            #df['threashold'] = th_vec
            #df.to_csv(self.th_file, index=False)
        
    #-------------------------------------------------------------------
    def load_threashold_file(self):
        if os.path.exists(self.th_file):
            #df = pd.read_csv(self.th_file, header=0)
            th_file = open(self.th_file, 'r')
            self.threashold = []
            for line in th_file:
                self.threashold.append([float(line.strip())])
            '''n_class = df.shape[0]
            th_vec = df['threashold']
            for i in range(n_class):
                self.threashold.append([th_vec[i]])'''

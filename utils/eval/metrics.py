import numpy as np
import math
from abc import abstractclassmethod
import torch
from torch import tensor
from torchmetrics.classification import BinaryHammingDistance, BinaryAveragePrecision
class Metrics:
    def __init__(self,truth,pred,c=1,dtype=np.int64):
        self.pred_prob=pred
        self.truth = np.equal(truth,c)
        self.pred = np.equal(pred,c)
        self.c=c
        self.dtype=dtype

###############################
#        Pixel Based         #
###############################

# Confusion Matrix
    
    def calc_ConfusionMatrix(self):
        # Obtain predicted and actual condition
        gt = np.equal(self.truth, self.c)
        pd = np.equal(self.pred, self.c)
        not_gt = np.logical_not(gt)
        not_pd = np.logical_not(pd)
        # Compute Confusion Matrix
        tp = np.logical_and(pd, gt).sum()
        tn = np.logical_and(not_pd, not_gt).sum()
        fp = np.logical_and(pd, not_gt).sum()
        fn = np.logical_and(not_pd, gt).sum()
        # Convert to desired numpy type to avoid overflow
        tp = tp.astype(self.dtype)
        tn = tn.astype(self.dtype)
        fp = fp.astype(self.dtype)
        fn = fn.astype(self.dtype)
        # Return Confusion Matrix
        return tp, tn, fp, fn
# True Positive
    def calc_TruePositive(self):
        # Obtain predicted and actual condition
        gt = np.equal(self.truth, self.c)
        pd = np.equal(self.pred, self.c)
        not_gt = np.logical_not(gt)
        not_pd = np.logical_not(pd)
        # Compute true positive
        tp = np.logical_and(pd, gt).sum()
        # Return true positive
        return tp
    
# True Negative 
    def calc_TrueNegative(self):
        # Obtain predicted and actual condition
        gt = np.equal(self.truth, self.c)
        pd = np.equal(self.pred, self.c)
        not_gt = np.logical_not(gt)
        not_pd = np.logical_not(pd)
        # Compute true negative
        tn = np.logical_and(not_pd, not_gt).sum()
        # Return true negative
        return tn

# False Positive
    def calc_FalsePositive(self):
        # Obtain predicted and actual condition
        gt = np.equal(self.truth, self.c)
        pd = np.equal(self.pred, self.c)
        not_gt = np.logical_not(gt)
        not_pd = np.logical_not(pd)
        # Compute false positive
        fp = np.logical_and(pd, not_gt).sum()
        # Return false positive
        return fp

# False Negative 
    def calc_FalseNegative(self):
        # Obtain predicted and actual condition
        gt = np.equal(self.truth, self.c)
        pd = np.equal(self.pred, self.c)
        not_gt = np.logical_not(gt)
        not_pd = np.logical_not(pd)
        # Compute false negative
        fn = np.logical_and(not_pd, gt).sum()
        # Return false negative
        return fn
# Accuracy
    def calc_Accuracy_CM(self):
        # Obtain confusion mat
        tp, tn, fp, fn = self.calc_ConfusionMatrix()
        # Calculate Accuracy
        acc = (tp + tn) / (tp + tn + fp + fn)
        # Return computed Accuracy
        return acc
# Precision
    def calc_Precision_CM(self):
        # Obtain confusion matrix
        tp, tn, fp, fn = self.calc_ConfusionMatrix()
        # Calculate precision
        if (tp + fp) != 0 : prec = (tp) / (tp + fp)
        else : prec = 0.0
        # Return precision
        return prec

# Recall - Sensitivity
    def calc_Sensitivity_CM(self):
        # Obtain confusion matrix
        tp, tn, fp, fn = self.calc_ConfusionMatrix()
        # Calculate sensitivity
        if (tp + fn) != 0 : sens = (tp) / (tp + fn)
        else : sens = 0.0
        # Return sensitivity
        return sens
    
# Specifity
    def calc_Specificity_CM(self):
        # Obtain confusion matrix
        tp, tn, fp, fn = self.calc_ConfusionMatrix()
        # Calculate specificity
        if (tn + fp) != 0 : spec = (tn) / (tn + fp)
        else : spec = 0.0
        # Return specificity
        return spec
# F-Measure (F1-score)
    def calc_F1_score(self):
        # Obtain precision and recall
        pres=self.calc_Precision_CM()
        reca=self.calc_Sensitivity_CM()
        # Calculate f1 score
        f1=2*((pres*reca)/(pres+reca))
        return f1
# F-Measure (F2-score)
    def calc_F2_score(self,beta):
        # Obtain precision and recall
        pres=self.calc_Precision_CM()
        reca=self.calc_Sensitivity_CM()
        # Calculate f1 score
        f2=(1+beta**2)*((pres*reca)/((beta*pres)+reca))
        return f2
# Dice Coefficient
    def calc_DSC_CM(self):
        # Obtain confusion mat
        tp, tn, fp, fn = self.calc_ConfusionMatrix()
        # Calculate Dice
        if (2*tp + fp + fn) != 0 : dice = 2*tp / (2*tp + fp + fn)
        else : dice = 0.0
        # Return computed Dice
        return dice
# Jaccard Index (IoU)
    def calc_IoU_CM(self):
        # Obtain confusion mat
        tp, tn, fp, fn = self.calc_ConfusionMatrix()
        # Calculate IoU
        if (tp + fp + fn) != 0 : iou = tp / (tp + fp + fn)
        else : iou = 0.0
        # Return computed IoU
        return iou
# Tversky Index
    def calc_Tversky_CM(self,alpha,beta):
        # Obtain confusion mat
        tp, tn, fp, fn = self.calc_ConfusionMatrix()
        # Calculate Dice
        if (tp+(alpha*fp)+(beta*fn)) != 0 : tver = tp/(tp+(alpha*fp)+(beta*fn))
        else : tver = 0.0
        # Return computed Dice
        return tver
# Area under the curve (AUC) - Receiver Operating Characteristics (ROC)
    def calc_AUC_trapezoid(self):
        # Obtain confusion mat
        tp, tn, fp, fn = self.calc_ConfusionMatrix()
        # Compute AUC
        if (fp+tn) != 0 : x = fp/(fp+tn)
        else : x = 0.0
        if (fn+tp) != 0 : y = fn/(fn+tp)
        else : y = 0.0
        auc = 1 - (1/2)*(x + y)
        # Return AUC
        return auc
    
# Matthews Correlation Coefficient (MCC)
    def calc_MCC(self):
        tp, tn, fp, fn = self.calc_ConfusionMatrix()
        # Verify if we need an approximation to zero (prove in reference)
        if (tp >= 1 and fp >= 1 and tn == 0 and fn == 0) or \
            (tp == 0 and fp >= 1 and tn >= 1 and fn == 0) or \
            (tp == 0 and fp == 0 and tn >= 1 and fn >= 1) or \
            (tp >= 1 and fp == 0 and tn == 0 and fn >= 1):
            return 0.0
        # else compute mcc
        top = tp*tn - fp*fn
        bot_raw = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
        bot = math.sqrt(bot_raw)
        if bot != 0 : mcc = top / bot
        else : mcc = 0.0
        # Return mcc score
        return mcc
    
###############################
#        Region Based         #
###############################

# Hamming Distance
    def calc_Hamm(self):
        truth = tensor(self.truth)
        pred = tensor(self.pred)
        metric = BinaryHammingDistance()
        return metric(truth, pred)

# Error Function
    def __E(self,p):
        # Define version of image that only retains the set of interest
        R1  = np.zeros(self.truth.shape).astype(bool)
        R2c = np.zeros(self.pred.shape).astype(bool)
        R1[np.where(self.truth == self.truth[p[0],p[1]])]  = True # Only keep set that contains the pixel at p
        R2c[np.where(self.pred != self.pred[p[0],p[1]])] = True # Only keep complement of set that contains pixel at p
        
        # Calculate numerator and denominator of Local Error
        num = np.count_nonzero(np.logical_and(R1,R2c))
        den = np.count_nonzero(R1)
        
        # Return Local Error
        return float(num)/den
# Local Consistency Error (LCE)
    def calc_LCE(self):
        if not np.issubdtype(self.truth.dtype, np.bool) or not np.issubdtype(self.pred.dtype, np.bool):
            print("⚠️ The arrays contain non-integer values. Use 0, 1, or integer class labels.")
            return None
        Eout = 0
        n = self.truth.shape[0] * self.truth.shape[1]

        for i in range(self.truth.shape[0]):
            for j in range(self.truth.shape[1]):
                E1 = self.__E([i, j])  # truth → pred

                # Swap truth and pred for the reverse direction
                truth_temp, pred_temp = self.truth, self.pred_prob
                self.truth, self.pred_prob = pred_temp, truth_temp
                E2 = self.__E([i, j])  # pred → truth
                self.truth, self.pred_prob = truth_temp, pred_temp  # Restore

                Eout += np.min([E1, E2])

        return Eout / n
# Global Consistency Error (GCE)
    def calc_GCE(self):
        if not np.issubdtype(self.truth.dtype, np.bool) or not np.issubdtype(self.pred.dtype, np.bool):
            print("⚠️ The arrays contain non-integer values. Use 0, 1, or integer class labels.")
            return None
        # Prepare for Loop
        E1 = 0
        E2 = 0
        n  = len(self.truth) * len(self.truth[0])
        
        for i in range(len(self.truth)):
            for j in range(len(self.truth[0])):
                E1 += self.__E([i,j])
                E2 += self.__E([i,j])
                

        return 1.0/n * np.min([E1,E2])
    
# Bidirectional Consistency Error (BCE)
# Average Precision 
    def calc_AverPres(self):
        truth = tensor(self.truth)  
        pred = tensor(self.pred_prob)  
        metric = BinaryAveragePrecision(thresholds=None)
        try:
            return metric(pred, truth)
        except:
            print("⚠️ Prediction array must contain probabilities (floats between 0 and 1).")
            return None

        

###############################
#    Author:Dominik Müller    #
###############################

#   Confusion Matrix
#   True Positive
#   True Negative 
#   False Positive
#   False Negative
#   Accuracy  
#   Precision
#   Recall / Sensitivity
#   Dice coefficient
#   Jaccard Index (IoU)
#   Area under the curve (AUC) - ROC
#   Matthews Correlation Coefficient (MCC)
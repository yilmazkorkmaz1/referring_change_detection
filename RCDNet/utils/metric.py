import numpy as np
import math
import scipy.stats as stats

np.seterr(divide='ignore', invalid='ignore')


def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))
    confusionMatrix = np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                        minlength=n_cl ** 2).reshape(n_cl, n_cl)
    return confusionMatrix, labeled, correct



def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa

def compute_semantic_scores(hist):
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0] #TN
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0] #FP. hist.sum(1): pred_hist
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0] #FN. hist.sum(0): label_hist
    c2hist[1][1] = hist_fg.sum() #TP
    #print('bn_hist: TP %d, FN %d, FP %d, TN %d'%(c2hist[1][1], c2hist[1][0], c2hist[0][1], c2hist[0][0]))
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3*IoU_mean + 0.7*Sek
    print('Mean IoU (semantic)= %.3f' % (IoU_mean*100))
    print('Sek = %.3f' % (Sek*100))
    print('Score = %.3f' % (Score*100))

    pixel_sum = hist.sum()
    change_pred_sum  = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    change_ratio = change_label_sum/pixel_sum
    SC_TP = np.diag(hist[1:, 1:]).sum()
    SC_Precision = SC_TP/change_pred_sum
    SC_Recall = SC_TP/change_label_sum
    semantic_f1 = stats.hmean([SC_Precision, SC_Recall])

    print('semantic_f1: %.4f' % (semantic_f1*100))
    
    return IoU_mean, Sek, Score, semantic_f1
    
   #print('change_ratio = %.4f, SC_Precision = %.4f, SC_Recall = %.4f, F_scd = %.4f' % (change_ratio*100, SC_Precision*100, SC_Recall*100, F1*100))

def compute_score(hist, correct, labeled):

    Semantic_IoU, Sek, Score, semantic_f1  = compute_semantic_scores(hist)

    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IoU = np.nanmean(iou)

    
    # Calculate recall and precision for all classes (excluding background)
    hist_no_back = hist[1:, 1:]  # Remove background class
    tp_sum = np.diag(hist_no_back).sum()  # Sum of true positives for all non-background classes
    recall = tp_sum / hist_no_back.sum(axis=1).sum()  # Total TP / (TP + FN)
    precision = tp_sum / hist_no_back.sum(axis=0).sum()  # Total TP / (TP + FP)
    
    # Original per-class metrics for class 1
    recall_1 = hist[1,1]/(hist[1].sum())
    precision_1 = hist[1,1]/(hist[:,1].sum())

    freq = hist.sum(1) / hist.sum()
    freq_IoU = (iou[freq > 0] * freq[freq > 0]).sum()

    classAcc = np.diag(hist) / hist.sum(axis=1)
    mean_pixel_acc = np.nanmean(classAcc)

    pixel_acc = correct / labeled

    return iou, Score, Sek, Semantic_IoU, semantic_f1, recall, precision, freq_IoU, mean_pixel_acc, pixel_acc
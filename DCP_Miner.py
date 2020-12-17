import os
import pandas as pd
from gensim.models import Word2Vec
from ast import literal_eval
import numpy as np
import gensim
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt
from scipy import interp
from statistics import mean


def get_features_for_Classification(data):

    data_nocode = data.No_code_tokens
    data_nocode = [literal_eval(x) for x in data_nocode]

    data_phraseList = data.ranked_phraseList
    data_phraseList_L = [literal_eval(x) for x in data_phraseList]


    feature_set = [a + b  for a, b,  in zip(data_nocode, data_phraseList_L)]

    return feature_set

# feature extraction from word2vec
def feature_extraction(data_input, word2vec_model):
    features = word_averaging_list(word2vec_model.wv, [sentence for sentence in data_input])
    return features
def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list])
def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif (word == 'NoneType'):
            print(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        return np.zeros(wv.vector_size, )

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def K_fold_SVC_classify_evaluate_and_print_classificationReprot(X, Y):
    nFolods = 10
    skf = StratifiedKFold(n_splits=nFolods, shuffle=True, random_state=1)
    lst_accu_stratified = []
    lst_precision_stratified = []
    lst_recall_stratified = []
    lst_F1_stratified = []
    tn_all, tp_all, fn_all, fp_all = 0,0,0,0
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    svc = SVC(C=10, gamma='scale', kernel='rbf', class_weight='balanced', max_iter=10000)

    for train_index, test_index in skf.split(X, Y):

        svc.fit(X[train_index], Y[train_index])

        # evaluate
        y_pred = svc.predict(X[test_index])

        tn, fp, fn, tp = confusion_matrix(Y[test_index], y_pred).ravel()
        accuracy_val = (tp + tn) / (tp + tn + fn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1Score = 2 * (precision * recall) / (precision + recall)

        #get stratified scores
        lst_accu_stratified.append(accuracy_val)
        lst_precision_stratified.append(precision)
        lst_recall_stratified.append(recall)
        lst_F1_stratified.append(F1Score)

        tn_all += tn
        tp_all += tp
        fn_all += fn
        fp_all += fp

        fpr, tpr, threshold = roc_curve(Y[test_index], y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    print('For SVC:  ')
    print('Strat Accuracy:', lst_accu_stratified)
    print('Strat Precision:', lst_precision_stratified)
    print('Strat Recall:', lst_recall_stratified)
    print('Strat F1:', lst_F1_stratified)

    print('Strat Overall Accuracy:', mean(lst_accu_stratified)*100)
    print('Strat Overall Precision:', mean(lst_precision_stratified)*100)
    print('Strat Overall Recall:', mean(lst_recall_stratified)*100)
    print('Strat Overall F1:', mean(lst_F1_stratified)*100)

    print('TN_all: ', (tn_all)/nFolods)
    print('TP_all: ', (tp_all)/nFolods)
    print('FN_all: ', (fn_all)/nFolods)
    print('FP_all: ', (fp_all)/nFolods)

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='navy', label=r'SVC Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

    return svc


def main():
    path_dir = os.path.dirname(
        '/Users/laks/Desktop/Mash-noSync/NewDownload/SO/DP_AP_For_QualityAtrributes/ContextMining/DP/EMSE/Exp/ADP-DC Miner/')

    read_miner = os.path.join(path_dir, 'EMSE_dataset4_for_miner_3.csv')
    read_classifier_data = os.path.join(path_dir, 'EMSE_dataset4_for_classifier.csv')

    #Load the trained RL model with Collocations
    RL_model = Word2Vec.load('/Users/laks/Desktop/Mash-noSync/NewDownload/SO/DP_AP_For_QualityAtrributes/ContextMining/DP/DataFiles/V5_1/w2v_allADP.model')


    '''Supervised Learning'''
    classi_data = pd.read_csv(read_classifier_data, sep='\t')
    # Select Featrues for classificaiton
    classification_data_input = get_features_for_Classification(classi_data)
    clasi_data_features = feature_extraction(classification_data_input, RL_model)
    #get the labels
    labels = classi_data.ContextTag.astype('category')
    clasi_coded_labels = np.asarray(labels.cat.codes)
    miner = K_fold_SVC_classify_evaluate_and_print_classificationReprot(clasi_data_features, clasi_coded_labels)

    '''Mining for more data'''
    miner_data = pd.read_csv(read_miner, sep='\t')
    minder_data_input = get_features_for_Classification(miner_data)
    miner_data_features = feature_extraction(minder_data_input, RL_model)

    '''Get the miner prediction to label manually'''
    miner_predictions = miner.predict(miner_data_features)

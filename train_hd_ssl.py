import preprocessing
import numpy as np
import os
import torch
from sklearn import metrics
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt
import sslmodel
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader
import json
from sklearn.metrics import roc_curve
import argparse

parser = argparse.ArgumentParser(description='hd_gait_detection_with_SSL')

parser.add_argument(
    '--cohort',
    type=str,
    default='hd',
    help='hd-huntington patients only/hc-healthy control only/mixed/pd_owly - pd data from owlytics smartwatch')

parser.add_argument(
    '--gait-only',
    action='store_true',
    default=True,
    help='if False multilabel of gait and chorea mode and if True gait only mode')

parser.add_argument(
    '--preprocess-mode',
    action='store_true',
    default=False,
    help='if true the raw data is preprocessed')

parser.add_argument(
    '--initialize-model',
    action='store_true',
    default=False,
    help='if true training and validation of the model is applied')

parser.add_argument(
    '--training-mode',
    action='store_true',
    default=False,
    help='if true training of the model is applied')

parser.add_argument(
    '--eval-mode',
    action='store_true',
    default=False,
    help='if true evaluation of training results is applied')

parser.add_argument(
    '--run-suffix',
    type=str,
    default='5sec_all',
    help='specify the run name')

parser.add_argument(
    '--wandb-flag',
    action='store_true',
    default=False,
    help='if true log to wandb')

parser.add_argument(
    '--model-type',
    type=str,
    default='segmentation',
    help='specify if segmentation or classification or vanila which is classification model trained on hc and evaluated on hd')

parser.add_argument(
    '--padding-type',
    type=str,
    default='triple_wind',
    help='specify if without_edges or triple_wind')

parser.add_argument(
    '--data-type',
    type=str,
    default='in_lab',
    help='specify if in_lab or daily_living; pd without fine-tuning is daily living and with is in-lab, vanila is under daily-living')

parser.add_argument(
    '--dataset-hd',
    type=str,
    default='iwear',
    help='specify if iwear or pace')



args = parser.parse_args()


VISUALIZE_ACC_VS_PRED_WIN = False
RAW_DATA_AND_LABELS_DIR = '/home/dafnas1/datasets/hd_dataset/lab_geneactive/synced_labeled_data_walking_non_walking'
#RAW_DATA_AND_LABELS_DIR = '/mlwell-data2/dafna/daily_living_data_array/HC'
#RAW_DATA_AND_LABELS_DIR = '/mlwell-data2/dafna/PD_data_and_labels/Data'
#PD_RAW_LABELS_DIR = '/mlwell-data2/dafna/PD_data_and_labels/labels'
#RAW_DATA_AND_LABELS_DIR = '/mlwell-data2/dafna/daily_living_data_array/PACE'
PROCESSED_DATA_DIR ='/mlwell-data2/dafna/daily_living_data_array/data_ready'
OUTPUT_DIR = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs'
VIZUALIZE_DIR = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/multiclass_hd_only/multiclass_separated_labels'
if args.cohort == 'pd_owly':
    SRC_SAMPLE_RATE = int(25) #hz
else:
    SRC_SAMPLE_RATE = int(100) #hz
STD_THRESH = 0.1
WINDOW_SIZE = int(30*10)
if args.model_type=='classification':
    WINDOW_OVERLAP=0
elif args.padding_type=='triple_wind':
    WINDOW_OVERLAP = int(30*5)
elif args.padding_type=='without_edges':
    WINDOW_OVERLAP = int(30*4)
else:
    WINDOW_OVERLAP=0
class GaitChoreaBaseEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 batch_size=64, 
                 device='cpu',
                 weights_path='state_dict.pt',
                 load_from_weights_path=False,
                 is_init_estimator = True) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.weights_path=weights_path
        self.is_init_estimator = is_init_estimator
        self.repo_tag = 'v1.0.0'
        self.model = self._get_model(pretrained=True)
        if load_from_weights_path and EXTERNAL_MODEL_FILE is not None: 
            checkpoint = torch.load(EXTERNAL_MODEL_FILE)
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.verbose=True

    def fit(self, x, y, sample_weight=None):
        train_dataset = sslmodel.NormalDataset(x, y, pid=None, name="training",augmentation=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )
        val_loader = None       

        sslmodel.train(self.model, 
                       train_loader, 
                       val_loader, 
                       self.device, 
                       model_type=model_type,
                       gait_only=GAIT_ONLY,
                       class_weights=None, 
                       weights_path=self.weights_path,
                       wandb_flag=False, 
                       is_init_estimator=self.is_init_estimator)
        return self

    def predict(self, X):
        dataset = sslmodel.NormalDataset(X, name='prediction')
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )
        
        _, y_logits,y_pred, _ = sslmodel.predict(self.model, dataloader, self.device)

        return y_logits

    def _get_model(self, pretrained):
        return sslmodel.get_sslnet(tag=self.repo_tag, pretrained=pretrained, num_classes=7,model_type=model_type,padding_type=padding_type)
    

def train_multiclass(X_train, y_train, X_test=None, y_test=None, batch_size=64, device='cpu', weights_path='', skip_train=False):
    init_estimator=GaitChoreaBaseEstimator(batch_size=batch_size, 
                                           device=device, 
                                           weights_path=weights_path, 
                                           load_from_weights_path=skip_train, 
                                           is_init_estimator=True)
    if skip_train:
        assert n_estimators==0, 'skip_train is supported only if n_estimators is 0'
        return [init_estimator]
    init_estimator.fit(X_train, y_train)
    y_pred = init_estimator.predict(X_train)
    estimators = [init_estimator]

    acc_gait, acc_chorea = sslmodel.calc_gait_and_chorea_acc(torch.Tensor(y_train), torch.Tensor(y_pred))
    wandb_log({'init_train_acc_gait':acc_gait, 'init_train_acc_chorea':acc_chorea})
    # print(f'init_gait_acc: {acc_gait},  init_chorea_acc: {acc_chorea}')
    if X_test is not None and y_test is not None:
        test_y_pred = init_estimator.predict(X_test)
        test_acc_gait, test_acc_chorea = sslmodel.calc_gait_and_chorea_acc(torch.Tensor(y_test), torch.Tensor(test_y_pred))
        # print(f'Test gait_acc: {test_acc_gait},  Test chorea_acc: {test_acc_chorea}')
        wandb_log({'init_test_acc_gait':test_acc_gait, 'init_test_acc_chorea':test_acc_chorea})

def groupkfold(groups, n_splits=5):
    """ Like GroupKFold but ordered """

    ord_unq_grps = groups[np.sort(np.unique(groups, return_index=True)[1])]
    folds_unq_grps = np.array_split(ord_unq_grps, n_splits)

    for unq_grps in folds_unq_grps:
        mask = np.isin(groups, unq_grps)
        test_idxs = np.nonzero(mask)
        train_idxs = np.nonzero(~mask)
        yield train_idxs, test_idxs

def wandb_log(dict_to_log):
    if args.wandb_flag:
        wandb.log(dict_to_log)
    print(dict_to_log)

def predict_boosting(x,estimators):
    y_pred = estimators[0].predict(x)
    for estimator in estimators[1:]:
        y_pred = update_y_pred(y_pred,estimator.predict(x))
    return y_pred

def get_valid_chorea(y_train):
    return np.sum(y_train[:,:10], axis=1, keepdims=True)

def get_scores_for_gait_detection(y_true,y_pred,model_type):
    # if model_type=='vanila':

    #     gait_labels = sslmodel.get_gait(torch.Tensor(y_true),is_pred=False)
    if model_type in ['classification', 'vanila']:
        pred_y_gait = sslmodel.get_gait(torch.Tensor(y_pred),is_pred=True)   
        gait_labels = sslmodel.get_gait(torch.Tensor(y_true),is_pred=False)
    elif model_type=='segmentation':
        pred_y_gait = y_pred[:, 0:2, :]
        gait_labels = y_true[:, :, 0]
    gait_predictions = np.argmax(pred_y_gait,axis=1)
    gait_predictions_logits = pred_y_gait
    return gait_predictions,gait_predictions_logits,gait_labels

def get_scores_for_chorea_detection(y_true,y_pred,model_type):
    if model_type in ['classification', 'vanila']:
        pred_y_chorea = sslmodel.get_chorea(torch.Tensor(y_pred), is_pred=True)   
        chorea_labels = sslmodel.get_chorea(torch.Tensor(y_true), is_pred=False)
    elif model_type=='segmentation':
        pred_y_chorea = y_pred[:, 2:7, :]   
        chorea_labels = y_true[:, :, 1]
    chorea_predictions = np.argmax(pred_y_chorea,axis=1)
    chorea_predictions_logits = pred_y_chorea
    return chorea_predictions,chorea_predictions_logits,chorea_labels

def auc_and_ci(labels,probs):
    
    # labels = labels.flatten()
    # prediction_exp = np.exp(predictions)
    # prob = prediction_exp[:,1,:] / np.sum(prediction_exp, axis=1)
    # prob = prob.flatten()
    # valid = valid.flatten()
    # valid_labels = labels[valid==1]
    # valid_prob = prob[valid==1]
    try:
        Y = probs[labels==1]
        X = probs[labels==0]
        n_x = len(X)
        n_y = len(Y)
        Y_exp = np.expand_dims(Y,axis=0)
        X_exp = np.expand_dims(X,axis=1)
        ge = Y_exp > X_exp
        eq = Y_exp == X_exp
        theta = (np.sum(ge) + 0.5*np.sum(eq))/(n_x * n_y + 1e-10)
        print(f"theta:{theta}")
        q1 = np.sum((np.sum(ge, axis=1) + 0.5*np.sum(eq, axis=1))**2)/(n_x*n_y**2 + 1e-10)
        q2 = np.sum((np.sum(ge, axis=0) + 0.5*np.sum(eq, axis=0))**2)/(n_y*n_x**2 + 1e-10)
        print(f"Q1:{q1}, Q2:{q2}")
        equal_prob = np.sum(eq)/(n_x * n_y + 1e-10)
        var = 1/(n_y-1)/(n_x-1)*(theta*(1-theta) - 1/4*equal_prob + (n_y-1)*(q1-theta**2) + (n_x-1)*(q2-theta**2))
        z=1.96
        ci = z*np.sqrt(var)
        print(f"var:{var}")
    except:
        theta = None
        ci = None
    
    # compare to auc sklearn
    try:
        sklearn_auc = metrics.roc_auc_score(labels, probs)
        print(f"sklearn_auc:{sklearn_auc}")
        precision, recall, _ = metrics.precision_recall_curve(labels, probs)
        pr_auc = metrics.auc(recall, precision)
        print(f"AUC of Precision-Recall Curve: {pr_auc}")
    except:
        pass
    return theta, ci

def auc_and_ci_per_chorea_lvl(gait_logits, gait_labels, chorea_labels, valid_chorea, valid_gait,analysis_type='per_pixel'):
    gait_logits_exp = np.exp(gait_logits)
    if model_type == 'segmentation':
        gait_prob = gait_logits_exp[:,1,:] / np.sum(gait_logits_exp, axis=1)
        if analysis_type == 'per_window':
            gait_prob, gait_labels_ind, chorea_labels_ind, valid_chorea = windowing(gait_prob, gait_labels, chorea_labels, valid_chorea, valid_gait)
        if analysis_type == 'per_pixel':
            valid_gait_f = valid_gait.flatten()
            valid_gait_ind = np.where(valid_gait_f)[0]
            gait_prob = gait_prob.flatten()[valid_gait_ind]
            gait_labels_ind = gait_labels.flatten()[valid_gait_ind]
            chorea_labels_ind = chorea_labels.flatten()[valid_gait_ind]
            valid_chorea = valid_chorea.flatten()[valid_gait_ind]
    else:
        gait_prob = gait_logits_exp[:,1] / np.sum(gait_logits_exp, axis=1)
        gait_labels_ind = torch.argmax(torch.Tensor(gait_labels), dim=-1)
        chorea_labels_ind = torch.argmax(torch.Tensor(chorea_labels), dim=-1)
    auc_dict = {}
    for is_valid in [0, 1]:
        valid_ind = np.where(valid_chorea == is_valid)[0]
        if is_valid:
            for chorea_level in np.unique(chorea_labels_ind):
                indices = np.where((chorea_labels_ind==chorea_level).flatten() * (valid_chorea == is_valid).flatten())[0]
                gait_prob_sel = gait_prob[indices]
                gait_labels_sel = gait_labels_ind[indices]
                if len(gait_labels_sel) > 0:
                    if analysis_type == 'per_pixel':
                        prefix2=f'{chorea_level}_per_pixel'
                    else:
                        prefix2=f'{chorea_level}'
                    _update_dict_res(auc_dict, prefix2, gait_labels_sel, gait_prob_sel)
            all_prefix = 'all_chorea'
            all_prefix = all_prefix + ('_per_pixel' if analysis_type == 'per_pixel' else '')
            _update_dict_res(auc_dict, all_prefix, gait_labels_ind, gait_prob)
            auc, ci = auc_and_ci(gait_labels_sel,gait_prob_sel) #change to logits
            cm = metrics.confusion_matrix(gait_labels_sel,gait_prob_sel>0.5)
            auc_dict[prefix2] = {'auc':auc,
                                 'ci':ci}
        else:
            gait_prob_sel = gait_prob[valid_ind]
            gait_labels_sel = gait_labels_ind[valid_ind]
            prefix1 = f''
            if analysis_type == 'per_pixel':
                prefix2=f'no_valid_chorea_per_pixel'
            else:
                prefix2=f'no_valid_chorea'
                _update_dict_res(auc_dict, prefix2, gait_labels_sel, gait_prob_sel)
                auc, ci = auc_and_ci(gait_labels_sel,gait_prob_sel) #change to logits
                auc_dict[prefix2] = {'auc':auc,
                                     'ci':ci}   
    return auc_dict

def _update_dict_res(res_dict, key, labels, prob):
    print(key)
    auc, ci = auc_and_ci(labels,prob) #change to logits
    sklearn_auc = metrics.roc_auc_score(labels, prob)
    precision, recall, _ = metrics.precision_recall_curve(labels, prob)
    pr_auc = metrics.auc(recall, precision)
    #plot_curves(labels, prob)
    cm = metrics.confusion_matrix(labels,prob>0.5)
    recall = cm[1,1] / (cm[1,0] + cm[1,1] + 1e-10)
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    res_dict[key] = {'auc':auc,
                     'ci':ci,
                     'recall':recall,
                     'precision': precision,
                     'f1_score':f1_score,
                     'sklearn-roc-auc': sklearn_auc,
                     'PR-RC auc':pr_auc
                     }

    def update_scores_file(scores_file, new_config_name, new_scores):
    try:
        # Load existing scores from file
        with open(scores_file, 'r') as f:
            existing_scores = json.load(f)
    except FileNotFoundError:
        existing_scores = {}

    # Update existing scores with new configuration
    existing_scores[new_config_name] = new_scores

    # Save updated scores back to the file
    with open(scores_file, 'w') as f:
        json.dump(existing_scores, f, indent=4)

def find_optimal_threshold_roc(pred_probs, true_labels,subject_name,plot_roc=False,output_dir=None):
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    
    # Calculate Youden's J statistic
    youden_j = tpr - fpr
    optimal_threshold_index = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_threshold_index]
    roc_auc = metrics.auc(fpr, tpr)
    if plot_roc:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], color='red', marker='o', label=f'Optimal threshold = {optimal_threshold:.2f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic_{subject_name}_optimal_thr')
        plt.legend(loc="lower right")
        output_dir = VIZUALIZE_DIR if output_dir is None else output_dir
        plt.savefig(os.path.join(output_dir,f'Receiver Operating Characteristic_{subject_name}.png'))
    return optimal_threshold

def plot_curves(labels, probs):
    fpr, tpr, _ = metrics.roc_curve(labels, probs)
    precision, recall, _ = metrics.precision_recall_curve(labels, probs)
    np.save(os.path.join('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves',f'fpr_{OUT_PREFIX}.npy'),fpr)
    np.save(os.path.join('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves',f'tpr_{OUT_PREFIX}.npy'),tpr)
    np.save(os.path.join('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves',f'precision_{OUT_PREFIX}.npy'),precision)
    np.save(os.path.join('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves',f'recall_{OUT_PREFIX}.npy'),recall)

    ipdb.set_trace()
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    # Save the ROC curve
    plt.savefig(os.path.join(VIZUALIZE_DIR,'roc_curve.png'))

    # Optionally, show the plot if you also want to visualize it
    plt.show()

    # Plot the Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % metrics.auc(recall, precision))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    # Save the Precision-Recall curve
    plt.savefig(os.path.join(VIZUALIZE_DIR,'pr_curve.png'))

def illustrate_results(cv_test_idxs_all_folds, win_subjects, win_acc_data, gait_predictions_all_folds, gait_labels_all_folds,
                    chorea_predictions_all_folds, chorea_labels_all_folds, valid_chorea_all_folds, valid_gait_all_folds,win_video_time):
    try:
        for fold_num, fold_ind in enumerate(cv_test_idxs_all_folds):
            for index_in_fold, ind in enumerate(fold_ind[0]):
                acc_data = win_acc_data[ind]
                subject = win_subjects[ind]
                video_time = win_video_time[ind]
                pred = gait_predictions_all_folds[ind]
                label = gait_labels_all_folds[ind]
                chorea_label = chorea_labels_all_folds[ind]
                valid_gait = valid_gait_all_folds[ind]
                acc_power = np.sqrt(np.mean(acc_data**2, axis=-1))
                acc_power_norm = acc_power/np.max(acc_power)
                plt.plot(acc_power_norm)
                plt.plot(label*0.9)
                plt.plot(pred)
                plt.plot(valid_gait)
                plt.plot(chorea_label/4)
                walking_ratio = np.sum(label*valid_gait)/(np.sum(valid_gait) + 1e-6)
                agreement_ratio = np.sum((pred==label)*valid_gait)/(np.sum(valid_gait) + 1e-6)
                plt.legend(['acc', 'label', 'pred', 'valid_gait','chorea_label'])
                plt.title(f'agreement ratio {agreement_ratio:.2f}')
                path_to_save = os.path.join(VIZUALIZE_DIR, "segmentation_visualize", f'{ind}_{subject}_{walking_ratio:.2f}_time_in_video_{video_time[0]:.2f}.jpg')
                print(f'saving {path_to_save}')
                plt.savefig(path_to_save)
                plt.close("all")
    except:
        ipdb.set_trace()

def windowing(gait_predictions, gait_labels, chorea_labels, valid_chorea, valid_gait,pred_or_prob='pred'):
    valid_gait_label = (gait_labels==1) * valid_gait
    valid_not_gait_label = (gait_labels==0) * valid_gait
    gait_windows = np.mean(valid_gait_label, axis=-1) > 0.6
    not_gait_windows = np.mean(valid_not_gait_label, axis=-1) > 0.7
    valid_indows = np.logical_or(gait_windows, not_gait_windows)
    indices = np.where(valid_indows)[0]
    if pred_or_prob == 'pred':
        new_gait_prediction = np.mean(gait_predictions[indices, :], axis=-1) > 0.5
    else:
        new_gait_prediction = np.mean(gait_predictions[indices, :], axis=-1)
    new_gait_labels = gait_windows[indices]
    new_valid_chorea = np.mean(valid_chorea[indices, :], axis=-1) > 0.5
    new_chorea_labels = np.round(np.sum(chorea_labels[indices, :] * valid_chorea[indices, :], axis=-1)/(np.sum(valid_chorea[indices, :], axis=-1)+1e-6))
    new_chorea_labels = np.ceil(np.sum(chorea_labels[indices, :] * valid_chorea[indices, :], axis=-1)/(np.sum(valid_chorea[indices, :], axis=-1)+1e-6))
    return new_gait_prediction, new_gait_labels, new_chorea_labels, new_valid_chorea
    

def confusion_matrix(labels, predictions, prefix1='', prefix2='', output_dir=None):
    if len(labels) == 0:
        return
    cm = metrics.confusion_matrix(labels, predictions) 
    try:
        recall = cm[1,1] / (cm[1,0] + cm[1,1] + 1e-10)
        precision = cm[1,1] / (cm[1,1] + cm[0,1] + 1e-10) if (cm[1,1] + cm[0,1]) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    except:
        # ipdb.set_trace()
        recall = 0
        precision = 0 
        f1_score = 0
    class_labels = ["Non-Walking", "Walking"]
    #plt.figure(figsize=(12, 12))
    ax = sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
        annot_kws={"fontsize": 16,"ha": "center", "va": "center"})
    ax.set_xticklabels(class_labels, fontsize=16, ha="center")
    ax.set_yticklabels(class_labels, fontsize=16, va="center") 
    
    

    plt.xlabel("Predicted",{"fontsize":16})
    plt.ylabel("True",{"fontsize":16})
    plt.title(f"Confusion Matrix Recall:{recall:.2f},Precision:{precision:.2f},F1:{f1_score:.2f}")
    output_dir = VIZUALIZE_DIR if output_dir is None else output_dir
    plt.savefig(os.path.join(output_dir,f'confusion_matrix_{prefix1}_{prefix2}.png'))
    plt.close('all')
    return recall

def generate_confusion_matrix_per_chorea_lvl(gait_predictions, gait_labels, chorea_predictions, chorea_labels, valid_chorea, valid_gait, fold_index,analysis_type='per_pixel', prefix=''):

    if model_type == 'segmentation':
        if analysis_type == 'per_window':
            gait_predictions, gait_labels_ind, chorea_labels_ind, valid_chorea = windowing(gait_predictions, gait_labels, chorea_labels, valid_chorea, valid_gait)
        if analysis_type == 'per_pixel':
            valid_gait_f = valid_gait.flatten()
            valid_gait_ind = np.where(valid_gait_f)[0]
            gait_predictions = gait_predictions.flatten()[valid_gait_ind]
            gait_labels_ind = gait_labels.flatten()[valid_gait_ind]
            chorea_labels_ind = chorea_labels.flatten()[valid_gait_ind]
            valid_chorea = valid_chorea.flatten()[valid_gait_ind]
    else:
        gait_labels_ind = torch.argmax(gait_labels, dim=-1)
        chorea_labels_ind = torch.argmax(chorea_labels, dim=-1)
    recall_dict = {}
    for is_valid in [0, 1]:
        valid_ind = np.where(valid_chorea == is_valid)[0]
        if is_valid:
            for chorea_level in np.unique(chorea_labels_ind):
                indices = np.where((chorea_labels_ind==chorea_level).flatten() * (valid_chorea == is_valid).flatten())[0]
                gait_predictions_sel = gait_predictions[indices]
                gait_labels_sel = gait_labels_ind[indices]
                if len(gait_labels_sel) > 0:
                    prefix1 = f'{fold_index}'
                    if analysis_type == 'per_pixel':
                        prefix2=f'{chorea_level}_per_pixel'
                    else:
                        prefix2=f'{chorea_level}'
                    recall = confusion_matrix(gait_labels_sel, gait_predictions_sel, prefix1=prefix1, prefix2=prefix2+prefix)
                    recall_dict[prefix2+prefix] = recall
        else:
            gait_predictions_sel = gait_predictions[valid_ind]
            gait_labels_sel = gait_labels_ind[valid_ind]
            prefix1 = f'{fold_index}'
            if analysis_type == 'per_pixel':
                prefix2=f'no_valid_chorea_per_pixel'
            else:
                prefix2=f'no_valid_chorea'
            recall_dict[prefix2+prefix] = confusion_matrix(gait_labels_sel, gait_predictions_sel, prefix1=prefix1, prefix2=f'no_valid_chorea'+prefix)
    return recall_dict

def main():
    if args.wandb_flag:
        wandb.init(project='hd_gait_detection_with_ssl')
    if args.preprocess_mode:
        #iterate over subjects and preprocess the data
        win_data_all_sub = np.empty((0,3,WINDOW_SIZE)) 
        if args.model_type == 'classification':
            win_labels_all_sub = win_subjects = all_subjects = win_chorea_all_sub = win_shift_all_sub = np.empty((0,1))
        elif  args.model_type == 'segmentation':
             win_labels_all_sub = win_subjects = all_subjects = win_chorea_all_sub = win_shift_all_sub = np.empty((0,WINDOW_SIZE))
        StdIndex_all = inclusion_idx = original_data_len = np.empty((0,))
        win_video_time_all_sub = np.empty((0,1))
        NumWin = []
        for file in os.listdir(RAW_DATA_AND_LABELS_DIR):
            try:
                if args.cohort == 'hc':
                    if 'TCCO' in file or 'CF' in file:
                        data_file = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file))
                    else:
                        continue
                if args.cohort == 'hd':
                    if 'TCCO' in file:
                        continue 
                    if 'WS' in file:
                        continue   
                    else:
                        data_file = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file))
                if args.cohort == 'pd_owly':
                    acc_data = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file),allow_pickle=True)
                    label_data = np.load(os.path.join(PD_RAW_LABELS_DIR, file.replace('data', 'labels')),allow_pickle=True)
                    data_file = {'arr_0': acc_data, 'arr_1': label_data}
                    
            except:
                print(f"cant open the file {file}")
                
                continue
            try:
                acc_data = data_file['arr_0'].astype('float')
            except:
                try:
                    # remove lines with empty string
                    acc_data = data_file['arr_0']
                    def is_numeric(s):
                        try:
                            float(s)
                            return True
                        except ValueError:
                            return False
                    # Convert non-numeric strings to np.nan
                    numeric_mask = np.array([[is_numeric(cell) for cell in row] for row in acc_data])
                    # numeric_mask = np.char.isnumeric(acc_data) | np.char.isdecimal(acc_data)
                    acc_data[~numeric_mask] = np.nan
                    acc_data = acc_data.astype('float')
                    acc_data = acc_data[~np.isnan(acc_data).any(axis=1)]
                    if len(acc_data) == 0:
                        continue
                except:  
                    print(f"failed to open {file}")                  
                    continue
            if args.cohort == 'pd_owly':
                labels = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file),allow_pickle=True)
            labels = data_file.get('arr_1', None)
            chorea = data_file.get('arr_2', None)
            if args.cohort == 'hc' and chorea is not None:
               chorea[chorea==-1] = 0 
            video_time = data_file.get('arr_3', None)
            subject_name = file.split('.')[0]
            
            if args.cohort == 'pd_owly':
                acc_data = preprocessing.highpass_filter(data=acc_data,high_cut=0.2,sampling_rate=SRC_SAMPLE_RATE,order=4)
            else:
                acc_data = preprocessing.bandpass_filter(data=acc_data,low_cut=0.2,high_cut=15,sampling_rate=SRC_SAMPLE_RATE,order=4)
            ## apply resampling 

            acc_data,labels, chorea, video_time = preprocessing.resample(data=acc_data,labels=labels,chorea=chorea, video_time=video_time ,original_fs=SRC_SAMPLE_RATE,target_fs=30)

            
            ## deivide data and labels to fixed windows
            data, labels, chorea, video_time, shift, NumWinSub = preprocessing.data_windowing(data=acc_data, labels=labels, chorea=chorea, video_time=video_time, window_size = WINDOW_SIZE, window_overlap=WINDOW_OVERLAP,
                                                                                std_th=STD_THRESH,model_type=args.model_type,subject=subject_name)
            # Concat the data and labels of the different subjects
            
            win_data_all_sub = np.append(win_data_all_sub, data, axis=0)
            win_labels_all_sub = np.append(win_labels_all_sub, labels, axis=0)
            win_chorea_all_sub = np.append(win_chorea_all_sub, chorea, axis=0)
            win_shift_all_sub = np.append(win_shift_all_sub, shift, axis=0)
            win_video_time_all_sub = np.append(win_video_time_all_sub, video_time, axis=0)
            print(file,win_data_all_sub.shape)
            # Create subject vector that will use for group the data in the training
            subject = np.tile(subject_name, (len(labels), 1)).reshape(-1, 1)
            win_subjects = np.append(win_subjects, subject)

            # StdIndex_all = np.append(StdIndex_all, StdIndex, axis=0)
            # inclusion_idx = np.append(inclusion_idx.squeeze(), inclusion, axis=0)
            # original_data_len = np.append(original_data_len, len(StdIndex))
            NumWin.append(NumWinSub)
         
        # Save the data, labels and groups
        res = {'win_data_all_sub': win_data_all_sub,
               'win_labels_all_sub': win_labels_all_sub,
               'win_subjects': win_subjects,
               'win_chorea_all_sub': win_chorea_all_sub,
               'win_shift_all_sub': win_shift_all_sub,
               'win_video_time_all_sub': win_video_time_all_sub
               }
    
        # create combined lable array of chorea and gait for multi-label model
        res = preprocessing.get_label_chorea_comb(res)
        np.savez(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model_{args.cohort}_only_{args.run_suffix}.npz'), **res)

            #old_data_files_included also:
            # res['arr_0'] = res['win_data_all_sub']
            # res['arr_1'] = res['win_labels_all_sub']
            # res['arr_2'] = res['win_subjects']
            # res['arr_5'] = res['win_chorea_all_sub']
            # res['arr_6'] = res['win_shift_all_sub']
            # res['arr_7'] = res['win_video_time_all_sub']
    if args.initialize_model:

        data_type = 'in_lab' # 'in_lab' or 'daily_living'  #pd without fine-tuning is  daily living and with is in-lab, vanila is under daily-living
        model_type = 'segmentation' # 'classification' or 'segmentation' or 'vanila'
        padding_type = 'triple_wind' # 'no_padding' or 'triple_wind' or 'without_edges'
        # !!!!remember to change segmentation/classifiction also in sslmodel.py !!!!!
        COHORT = 'hd' # hd or hc or pd_owly or hd_and_pd_train
        training_cohort = False #'hd_and_pd_train' 
        args.dataset_hd = 'iwear' # pace or iwear
        # as in the data file feom the preprocessing step
        
        #'segmentation_without_edges_overlap_final_1_4_24' #'segmentation_triple_wind_no_shift_7_4_24'
        #FILE_PREFIX = 'classification_test' #segmentation_val' or'segmentation_without_edges_overlap' or 'segmentation_triple_wind_no_shift'
        
        gait_only_prefix = "_gait_only" if args.gait_only else ""
        OUTPUT_DIR = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/output_files'
        #UTPUT_DIR = '/mlwell-data2/dafna/ssl_gait_detection/model_outputs/output_files'

        
        
        ## for in lab
        if args.data_type == 'in_lab':
            if args.cohort == 'hd':
                input_data_dir = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/data_ready' # /mlwell-data2/dafna/in_lab_data_array or /home/dafnas1/my_repo/hd_gait_detection_with_SSL/data_ready
                INP_PREFIX = 'segmentation_triple_wind_no_shift' #or 'check_labels'
                #segmentation_labels' or'segmentation_without_edges_overlap' or 'segmentation_triple_wind_no_shift' or 'classification_test' or 'daily_living_classification_full_files' or 'classification_hc'
                #as in the output file of preditions
                OUT_PREFIX = 'segmentation_triple_wind_no_shift_final_8_4_24' #'classification_test_final' or 'segmentation_triple_wind_no_shift_final_8_4_24'
            if args.cohort == 'hc':
                input_data_dir = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/data_ready' # or /mlwell-data2/dafna/in_lab_data_array
                INP_PREFIX = 'classification_hc'
                #as in the output file of preditions
                OUT_PREFIX = 'classification_7_4_24_chorea_0_only_gait_only_hc' 
            if args.cohort == 'pd_owly' and args.model_type=='classification':
                input_data_dir = '/mlwell-data2/dafna/daily_living_data_array/data_ready'
                INP_PREFIX = 'classification_with_std_rm'
                OUT_PREFIX = 'classification_with_std_rm_with_fine_tuning'
            if args.cohort == 'pd_owly' and args.model_type=='segmentation': 
                input_data_dir = '/mlwell-data2/dafna/daily_living_data_array/data_ready'
                INP_PREFIX = 'segmentation_triple_wind_with_std_rm'
                OUT_PREFIX = 'segmentation_triple_wind_with_std_rm_with_fine_tuning'
            if args.model_type == 'vanila':
                EXTERNAL_MODEL_FILE = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/output_files/multiclass_weights_hc_only_boosting_classification_7_4_24_chorea_0_only_gait_only.pt'
            else:
                EXTERNAL_MODEL_FILE = None
        ## for daily living
        if args.data_type == 'daily_living':
            input_data_dir = '/mlwell-data2/dafna/daily_living_data_array/data_ready'
            if args.dataset_hd == 'iwear':
                if args.cohort == 'hc' and args.model_type=='segmentation':
                    INP_PREFIX = 'daily_living_segmentation_triple_wind_full_files_no_std'
                    OUT_PREFIX = 'daily_living_seg_triple_wind_no_std'
                    EXTERNAL_MODEL_FILE = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/output_files/multiclass_weights_hc_only_boosting_segmentation_triple_wind_hc_7_4_24_chorea_0_only_gait_only.pt'
                if args.cohort == 'hc' and args.model_type=='classification':
                    INP_PREFIX = 'daily_living_classification_full_files_no_std'
                    OUT_PREFIX = 'daily_living_classification_no_std'
                    EXTERNAL_MODEL_FILE = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/output_files/multiclass_weights_hc_only_boosting_classification_7_4_24_chorea_0_only_gait_only.pt'
            if args.cohort == 'hd' and args.model_type=='segmentation': 
                if args.dataset_hd == 'iwear':
                    INP_PREFIX = 'daily_living_segmentation_triple_wind_full_files_no_std'
                    OUT_PREFIX = 'daily_living_seg_triple_wind_no_std'
                    EXTERNAL_MODEL_FILE = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/output_files/multiclass_weights_hd_only_boosting_segmentation_triple_wind_no_shift_final_8_4_24_gait_only.pt'
                elif args.dataset_hd == 'pace':
                    INP_PREFIX = 'segmentation_triple_wind_pace'
                    OUT_PREFIX = 'segmentation_triple_wind_daily_pace'
                    EXTERNAL_MODEL_FILE = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/output_files/multiclass_weights_hd_only_boosting_segmentation_triple_wind_no_shift_final_8_4_24_gait_only.pt'
            if args.cohort == 'hd' and args.model_type=='classification': 
                if args.dataset_hd == 'iwear':
                    INP_PREFIX = 'daily_living_classification_full_files_no_std'
                    OUT_PREFIX = 'daily_living_classification_no_std'
                    EXTERNAL_MODEL_FILE = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/output_files/multiclass_weights_hd_only_boosting_classification_test_final_gait_only.pt'
                elif args.dataset_hd== 'pace':
                    INP_PREFIX = 'classification_daily_pace'
                    OUT_PREFIX = 'classification_daily_pace'
                    EXTERNAL_MODEL_FILE = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/output_files/multiclass_weights_hd_only_boosting_classification_test_final_gait_only.pt'
            if args.cohort == 'pd_owly' and args.model_type=='segmentation': 
                INP_PREFIX = 'segmentation_triple_wind_with_std_rm'
                OUT_PREFIX = 'segmentation_triple_wind_with_std_rm'
                EXTERNAL_MODEL_FILE = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/output_files/multiclass_weights_hd_only_boosting_segmentation_triple_wind_no_shift_final_8_4_24_gait_only.pt'
            if args.cohort == 'pd_owly' and args.model_type=='classification':
                INP_PREFIX = 'classification_with_std_rm'
                OUT_PREFIX = 'classification_with_std_rm'
                EXTERNAL_MODEL_FILE = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/output_files/multiclass_weights_hd_only_boosting_classification_test_final_gait_only.pt' 
            if args.cohort == 'hd' and args.model_type=='vanila':
                # test the hc model on hd data
                input_data_dir = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/data_ready'
                INP_PREFIX = 'classification_test' 
                OUT_PREFIX = 'vanila_hc_model_on_hd' 
                EXTERNAL_MODEL_FILE = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/output_files/multiclass_weights_hc_only_boosting_classification_7_4_24_chorea_0_only_gait_only.pt'
                

        VIZUALIZE_DIR = f'/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/boosting/{OUT_PREFIX}'+gait_only_prefix+'_'+ args.cohort
        n_estimators = 0
        learning_rate = 0.5

        if not os.path.exists(VIZUALIZE_DIR):
            os.makedirs(VIZUALIZE_DIR)
            print(f"Directory '{VIZUALIZE_DIR}' created.")
        else:
            print(f"Directory '{VIZUALIZE_DIR}' already exists.")
        if args.training_mode:
            # if train in - lab data weights_path=weights_path if not load EXTERNAL_MODEL_FILE
            weights_path = os.path.join(OUTPUT_DIR,f'multiclass_weights_{args.cohort}_only_boosting_{OUT_PREFIX}'+gait_only_prefix+'.pt')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_class = 10
            
            
            input_file_name = f'windows_input_to_multiclass_model_{args.cohort}_only_{INP_PREFIX}.npz'
            print("start loading input file")
            input_file = np.load(os.path.join(input_data_dir,input_file_name))
            print("done loading input file")
            win_acc_data = input_file['win_data_all_sub']
            win_acc_data = np.transpose(win_acc_data,[0,2,1])
            win_subjects = groups =input_file['win_subjects']
            win_shift = input_file['win_shift_all_sub']
            print("done extract win acc")
            if args.model_type in ['classification', 'vanila']:
                win_labels = input_file['gait_label_chorea_comb']
                one_hot_labels = np.zeros((len(win_labels), num_class+2), dtype=int) # adding two extra dummies for no chorea
                one_hot_labels[np.arange(len(win_labels)), win_labels.squeeze().astype(int)] = 1
                win_labels_data = one_hot_labels
            elif args.model_type == 'segmentation':
                win_chorea = input_file['win_chorea_all_sub']
                win_labels = input_file['win_labels_all_sub']
                num_gait_class = 2  # 0 for non-walking, 1 for walking
                num_chorea_levels = 5  # 0 to 4
                win_shift = np.mean(win_shift, axis=-1)
                win_labels_data = np.stack([np.maximum(win_labels,0), np.maximum(win_chorea, 0), win_labels>=0, win_chorea>=0], axis=-1)
            
            print(f"done extract data for {args.model_type}")
            X_train = win_acc_data
            
            fold = 0
            gait_predictions_all_folds = []
            gait_predictions_logits_all_folds = []
            gait_labels_all_folds = []
            chorea_predictions_all_folds = []
            chorea_predictions_logits_all_folds = []
            chorea_labels_all_folds = []
            valid_chorea_all_folds = []
            valid_gait_all_folds = []

            cv_test_idxs_all_folds = []
            cv_train_idxs_all_folds = []
            
            if args.padding_type =='triple_wind':
                left_wind = win_acc_data[0:-2:2]
                mid_wind = win_acc_data[1:-1:2]
                right_wind =win_acc_data[2::2] 
                win_acc_data = np.concatenate([left_wind,mid_wind,right_wind],axis=1)
                win_labels_data = win_labels_data[1:-1:2]
                groups = groups[1:-1:2]
                win_subjects = win_subjects[1:-1:2]
            print("start folds generation")
            for train_idxs, test_idxs in groupkfold(groups, n_splits=5): 
                if args.wandb_flag:
                    run_num = 30
                    run = wandb.init(project='hd_gait_detection_with_ssl',
                            reinit=True,
                            group=f"experiment_{run_num}")
                    wandb.config.update({})
                    run_name = f'experiment_{run_num}_fold_{fold}'
                    wandb.run.name = run_name
                test_idxs = (test_idxs[0][np.where(win_shift[test_idxs[0]]==0)[0]],) # remove shift created in the preprocessing phase when != 0 from test
                X_train = win_acc_data[train_idxs]
                X_test = win_acc_data[test_idxs]
                y_train = win_labels_data[train_idxs]
                y_test = win_labels_data[test_idxs]
                
                cv_test_idxs = test_idxs
                cv_train_idxs = train_idxs
                print("start fitting model")
                skip_train = EXTERNAL_MODEL_FILE is not None or args.model_type=='vanila'
                estimators = train_multiclass(X_train, y_train, X_test, y_test, batch_size=64, device=device, weights_path=weights_path, skip_train=skip_train)
                print("start predicting fold")
                y_test_pred = predict_boosting(X_test,estimators)
                print(np.isnan(y_test_pred).any())
                gait_predictions,gait_predictions_logits,gait_labels = get_scores_for_gait_detection(y_test, y_test_pred, args.model_type)
                chorea_predictions,chorea_predictions_logits,chorea_labels = get_scores_for_chorea_detection(y_test, y_test_pred, args.model_type)
                if args.model_type in ['classification', 'vanila']:
                    valid_chorea = get_valid_chorea(y_test)
                    valid_gait = None
                if args.model_type == 'segmentation':
                    valid_chorea = y_test[:, :, 3]
                    valid_gait = y_test[:, :, 2]
                    
                test_acc_gait, test_acc_chorea = sslmodel.calc_gait_and_chorea_acc(torch.Tensor(y_test), torch.Tensor(y_test_pred))
                fold_index = len(gait_predictions_all_folds)
                generate_confusion_matrix_per_chorea_lvl(gait_predictions, gait_labels, chorea_predictions, chorea_labels, valid_chorea, valid_gait, fold_index)
                
                gait_predictions_all_folds.append(gait_predictions)
                gait_predictions_logits_all_folds.append(gait_predictions_logits)
                gait_labels_all_folds.append(gait_labels)
                valid_gait_all_folds.append(valid_gait)

                chorea_predictions_all_folds.append(chorea_predictions)
                chorea_predictions_logits_all_folds.append(chorea_predictions_logits)
                chorea_labels_all_folds.append(chorea_labels)
                valid_chorea_all_folds.append(valid_chorea)

                cv_test_idxs_all_folds.append(cv_test_idxs)
                cv_train_idxs_all_folds.append(cv_train_idxs)
                print(f'val_gait_acc_in_fold_{fold}: {test_acc_gait},  chorea_acc_in_fold_{fold}: {test_acc_chorea}')
                wandb_log({'gait_test_acc': test_acc_gait, 'chorea_test_acc': test_acc_chorea})
                if args.wandb_flag:
                    run.finish()
                
                fold+=1
            if args.model_type == 'segmentation':
                cat_func = np.concatenate
            else:
                cat_func = torch.cat
            gait_predictions_all_folds = cat_func(gait_predictions_all_folds)
            gait_predictions_logits_all_folds = cat_func(gait_predictions_logits_all_folds)  
            gait_labels_all_folds = cat_func(gait_labels_all_folds)
            if args.model_type == 'segmentation':
                valid_gait_all_folds = np.concatenate(valid_gait_all_folds)
            else:
                valid_gait_all_folds = None
            chorea_predictions_all_folds = cat_func(chorea_predictions_all_folds)
            chorea_predictions_logits_all_folds = cat_func(chorea_predictions_logits_all_folds)
            chorea_labels_all_folds = cat_func(chorea_labels_all_folds)
            valid_chorea_all_folds = np.concatenate(valid_chorea_all_folds)

            if True:
            generate_confusion_matrix_per_chorea_lvl(gait_predictions_all_folds, 
                                                    gait_labels_all_folds, 
                                                    chorea_predictions_all_folds, 
                                                    chorea_labels_all_folds, 
                                                    valid_chorea_all_folds, 
                                                    valid_gait_all_folds,
                                                    fold_index='all')
        

            np.savez(os.path.join(OUTPUT_DIR, f'multiclass_separated_labels_predictions_and_logits_with_true_labels_and_subjects_{COHORT}_only_boosting_{OUT_PREFIX}'+gait_only_prefix+'.npz'),
                        gait_predictions_all_folds=gait_predictions_all_folds,
                        gait_predictions_logits_all_folds=gait_predictions_logits_all_folds,
                        gait_labels_all_folds=gait_labels_all_folds,
                        chorea_predictions_all_folds=chorea_predictions_all_folds,
                        chorea_predictions_logits_all_folds=chorea_predictions_logits_all_folds,
                        chorea_labels_all_folds=chorea_labels_all_folds,
                        valid_chorea_all_folds=valid_chorea_all_folds,
                        valid_gait_all_folds=valid_gait_all_folds,
                        win_subjects=win_subjects, 
                        cv_test_idxs_all_folds=cv_test_idxs_all_folds)

        if args.evaluation_mode:
            # loading the output file to evaluate the results without training
            output_file = np.load(os.path.join(OUTPUT_DIR, f'multiclass_separated_labels_predictions_and_logits_with_true_labels_and_subjects_{COHORT}_only_boosting_{OUT_PREFIX}'+gait_only_prefix+'.npz'),allow_pickle=True)
            gait_predictions_all_folds = output_file['gait_predictions_all_folds'],
            gait_predictions_logits_all_folds = output_file['gait_predictions_logits_all_folds'],
            gait_labels_all_folds = output_file['gait_labels_all_folds'],
            chorea_predictions_all_folds = output_file['chorea_predictions_all_folds'],
            chorea_predictions_logits_all_folds = output_file['chorea_predictions_logits_all_folds'],
            chorea_labels_all_folds = output_file['chorea_labels_all_folds'],
            valid_chorea_all_folds = output_file['valid_chorea_all_folds'],
            valid_gait_all_folds = output_file['valid_gait_all_folds'],
            win_subjects = output_file['win_subjects'], 
            cv_test_idxs_all_folds = output_file['cv_test_idxs_all_folds']
            # debug start
            cv_test_idxs_all_folds_flat = np.concatenate([cv_test_idxs_all_folds[i][0] for i in range(len(cv_test_idxs_all_folds))])
            input_file_name = f'windows_input_to_multiclass_model_{COHORT}_only_{INP_PREFIX}.npz'
            print("start loading input file")
            input_file = np.load(os.path.join(input_data_dir,input_file_name))
            win_acc_data = input_file['win_data_all_sub']
            win_acc_data = np.transpose(win_acc_data,[0,2,1])
            win_video_time = input_file['win_video_time_all_sub']

            ## find mismatch between classification and segmentation
            if model_type == 'classification':
                counter = 0
                for i in range(len(gait_predictions_all_folds[0])):
                    if gait_predictions_all_folds[0][i]==0 and gait_labels_all_folds[0][i][1]==1:
                        print(counter, win_subjects[0][i], win_video_time[i])
                        counter +=1
                gait_labels = gait_labels_all_folds[0]  
                gait_predictions = gait_predictions_all_folds[0]  

                # Use np.where to find where the label is gait and the prediction is non-gait
                mismatch_indices = np.where((gait_labels[:, 1] == 1) & (gait_predictions == 0))[0]
                false_class_video_time = win_video_time[mismatch_indices]
                false_class_subject = win_subjects[0][mismatch_indices]
                
                np.savez('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/false_gait_classification_video_start_time_andsubjects.npz',
                false_class_video_time=false_class_video_time,false_class_subject=false_class_subject)
            #### compare classification and seg windows DEBUGG#####
            class_false_gait = dict(np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/false_gait_classification_video_start_time_andsubjects.npz'))
            class_false_gait_video = class_false_gait['false_class_video_time'].flatten()
            class_false_gait_subject = class_false_gait['false_class_subject'].flatten()
            for video, subject in zip(class_false_gait_video, class_false_gait_subject):
                sub_ind = np.where(win_subjects[0]==subject)[0]
                for ind in sub_ind:
                    if win_video_time[ind] >= video and win_video_time[ind]<=video+10 and ind%2==1:
                        try:
                            output_ind = (ind-1)//2
                            pred = gait_predictions_all_folds[0][output_ind]
                            label = gait_labels_all_folds[0][output_ind]
                            valid = valid_gait_all_folds[0][output_ind]
                            # get seg values
                            valid_gait_label = (label==1) * valid
                            valid_not_gait_label = (label==0) * valid
                            gait_windows = np.mean(valid_gait_label, axis=-1) > 0.6
                            not_gait_windows = np.mean(valid_not_gait_label, axis=-1) > 0.7
                            valid_windows = gait_windows or not_gait_windows
                            if not valid_windows:
                                continue
                            win_pred = np.mean(pred) > 0.5
                            print(f"subject:{subject}, index:{ind}, start_video:{video}, seg_start:{win_video_time[ind]}, is_label_gait:{gait_windows}, is_pred_gait:{win_pred}, agree:{gait_windows==win_pred}")
                        except:
                            ipdb.set_trace()

            ############# end debug ###############################
            ipdb.set_trace()
            optimal_th_sub_dict ={}
            ## get roc per patient
            # for subject_name in np.unique(win_subjects):
            #     print(subject_name)
            #     sub_indx=np.where(win_subjects[0]==subject_name)[0]
            #     sub_win_acc_data = win_acc_data[sub_indx]
            #     print(f"acc_shape_{sub_win_acc_data.shape}")
            #     sub_logits = gait_predictions_logits_all_folds[0][sub_indx]
            #     sub_exp = np.exp(sub_logits)
            #     sub_probs = sub_exp / (np.sum(sub_exp, axis=-1, keepdims=True) + 1e-7)
            #     sub_walking_prob = sub_probs[:,1]
            #     sub_labels = np.argmax(gait_labels_all_folds[0][sub_indx],axis=-1)
            #     optimal_th_sub = find_optimal_threshold_roc(sub_walking_prob, sub_labels,subject_name,plot_roc=True,output_dir=None)
            #     optimal_th_sub_dict[subject_name] = optimal_th_sub 
            #     print(f"Patient: {subject_name}, Optimal Threshold: {optimal_th_sub}")
            #     ipdb.set_trace()
            # ipdb.set_trace()
            auc_dict = auc_and_ci_per_chorea_lvl(gait_labels=gait_labels_all_folds[0],
                                                gait_logits=gait_predictions_logits_all_folds[0],
                                                valid_gait=valid_gait_all_folds[0],
                                                valid_chorea=valid_chorea_all_folds[0],
                                                chorea_labels=chorea_labels_all_folds[0],
                                                analysis_type='per_pixel')
            
            scores_file = os.path.join('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs','scores.json')

            # New model configuration name
            new_config_name = OUT_PREFIX+gait_only_prefix+'_'+COHORT
            update_scores_file(scores_file, new_config_name, auc_dict)
            
            ipdb.set_trace()

            # Update scores file with new configuration and scores
            

            if ILLUSTRATE_RESULTS:
                illustrate_results(cv_test_idxs_all_folds, win_subjects[0], win_acc_data, gait_predictions_all_folds[0], gait_labels_all_folds[0],
                            chorea_predictions_all_folds[0], chorea_labels_all_folds[0], valid_chorea_all_folds[0], valid_gait_all_folds[0],win_video_time)
            
            generate_confusion_matrix_per_chorea_lvl(gait_predictions_all_folds[0], 
                                                    gait_labels_all_folds[0], 
                                                    chorea_predictions_all_folds[0], 
                                                    chorea_labels_all_folds[0], 
                                                    valid_chorea_all_folds[0],
                                                    valid_gait_all_folds[0], 
                                                    fold_index='all',
                                                    analysis_type='per_pixel')

            all_recall_dict = {}
            for i in range(5):
                part_len = 300//5
                part_start = i*part_len
                part_end = (i+1)*part_len
                recall_dict = generate_confusion_matrix_per_chorea_lvl(gait_predictions_all_folds[0][:,part_start:part_end], 
                                                        gait_labels_all_folds[0][:,part_start:part_end], 
                                                        chorea_predictions_all_folds[0][:,part_start:part_end], 
                                                        chorea_labels_all_folds[0][:,part_start:part_end], 
                                                        valid_chorea_all_folds[0][:,part_start:part_end],
                                                        valid_gait_all_folds[0][:,part_start:part_end], 
                                                        fold_index='all',
                                                        analysis_type='per_pixel',
                                                        prefix=f'_part_{i}')
                all_recall_dict.update(recall_dict)
            for i in range(5):
                try:
                    recall = [all_recall_dict[f'{i}.0_per_pixel_part_{j}'] for j in range(5)]
                except:
                    continue # if there is no 1 
                print(recall)
                plt.plot(np.array(recall))
            plt.xlabel('part index')
            plt.ylabel('recall score')
            plt.legend(['chorea 0','chorea 1','chorea 2','chorea 3','chorea 4','chorea label not valid'])
            plt.savefig(os.path.join(VIZUALIZE_DIR, 'recall_per_part.jpg'))

            
            
            generate_confusion_matrix_per_chorea_lvl(gait_predictions_all_folds[0], 
                                                    gait_labels_all_folds[0], 
                                                    chorea_predictions_all_folds[0], 
                                                    chorea_labels_all_folds[0], 
                                                    valid_chorea_all_folds[0],
                                                    valid_gait_all_folds[0], 
                                                    fold_index='all',
                                                    analysis_type='per_pixel')
            ipdb.set_trace()
            
            ## error analysis per chorea level
            lvl3 = np.where(np.logical_and(np.logical_and(gait_predictions_all_folds[0]==0,valid_chorea_all_folds[0].flatten()==1), np.logical_and(np.argmax(chorea_labels_all_folds[0], axis=1)==3, np.argmax(gait_labels_all_folds[0], axis=1)==1)))[0]
            lvl3_index = cv_test_idxs_all_folds_flat[lvl3]
            lvl3_win = win_acc_data[lvl3_index]
            lvl3_win_video_time = win_video_time[lvl3_index]
            lvl3_win_subjects = win_subjects[0][lvl3_index]
            for sub, ti in zip(lvl3_win_subjects, lvl3_win_video_time):                                                  
                print(sub, ti//60, ti%60)
            
            lvl3_power = np.sum(lvl3_win * lvl3_win, axis=-1)
            for index in range(lvl3_power.shape[0]):
                plt.plot(lvl3_power[index])
                plt.savefig(os.path.join(VIZUALIZE_DIR,f'error_ana_power_plt_{index}.png'))
                plt.close('all')
            
            # debug end
if __name__ == '__main__':
    main()
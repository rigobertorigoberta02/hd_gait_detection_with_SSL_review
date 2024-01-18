import torch
import numpy as np
import sslmodel
import models_new
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader
from starboost import BoostingClassifier
import os
import ipdb
import wandb
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE





OUTPUT_DIR = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs'
VIZUALIZE_DIR = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/boosting/shifted_win_trn_std_bandpass_10sec_walk_th_corrected_labels'
n_estimators = 1
learning_rate = 0.5

is_multi_label = True
wandb_flag = True
TRAIN_MODE = True
EVAL_MODE = False

class GaitChoreaBaseEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 batch_size=64, 
                 device='cpu',
                 weights_path='state_dict.pt',
                 is_init_estimator = True) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.weights_path=weights_path
        self.is_init_estimator = is_init_estimator
        self.repo_tag = 'v1.0.0'
        self.model = self._get_model(pretrained=True)
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
                       class_weights=None, 
                       weights_path=self.weights_path,
                       wandb_flag=False, 
                       is_init_estimator=self.is_init_estimator)
        return self

    def predict(self, X):
        # sslmodel.verbose = self.verbose

        dataset = sslmodel.NormalDataset(X, name='prediction')
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )

        # model = self._get_model(pretrained=False)
        # # model.load_state_dict(self.state_dict)
        # model.to(self.device)
        _, y_logits,y_pred, _ = sslmodel.predict(self.model, dataloader, self.device)

        return y_logits

    def _get_model(self, pretrained):
        return sslmodel.get_sslnet(tag=self.repo_tag, pretrained=pretrained, num_classes=7)
    

def train_multiclass(X_train, y_train, X_test=None, y_test=None, batch_size=64, device='cpu', weights_path=''):
    # boosting_model = BoostingClassifier
    # gbgta = boosting_model(
    #     init_estimator=GaitChoreaBaseEstimator(
    #         batch_size=batch_size, 
    #         device=device,
    #         weights_path=weights_path),
    #     base_estimator=GaitChoreaBaseEstimator(
    #         batch_size=batch_size, 
    #         device=device,
    #         weights_path=weights_path),
    #     n_estimators=n_estimators,
    #     learning_rate=learning_rate)
    # # y = np.array(y_train)
    # # y = y.flatten()
    # gbgta.fit(X_train, y_train)
    init_estimator=GaitChoreaBaseEstimator(batch_size=batch_size, device=device, weights_path=weights_path, is_init_estimator=True)
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
    for n in range(n_estimators):
        ## equivalent to negative gradient 
        y_pred_proba = log_softmax(y_pred)
        y_residuals = calc_gradient(y_pred,y_train)
        ## add valid chorea flag to the residual
        y_residuals = np.concatenate([y_residuals, get_valid_chorea(y_train)], axis=1)
        estimator = GaitChoreaBaseEstimator(batch_size=batch_size, device=device,weights_path=weights_path, is_init_estimator=False)
        
        estimator.fit(X_train,y_residuals)
        
        ## equivalent to update terminal regions 

        y_pred = update_y_pred(y_pred,estimator.predict(X_train))
        acc_gait, acc_chorea = sslmodel.calc_gait_and_chorea_acc(torch.Tensor(y_train), torch.Tensor(y_pred))
        wandb_log({'train_acc_gait':acc_gait, 'train_acc_chorea':acc_chorea})
        # print(f'gait_acc: {acc_gait},  chorea_acc: {acc_chorea}')
        if X_test is not None and y_test is not None:
            test_y_pred = update_y_pred(test_y_pred,estimator.predict(X_test))
            test_acc_gait, test_acc_chorea = sslmodel.calc_gait_and_chorea_acc(torch.Tensor(y_test), torch.Tensor(test_y_pred))
            wandb_log({'test_acc_gait':test_acc_gait, 'test_acc_chorea':test_acc_chorea})
            # print(f'Test gait_acc: {test_acc_gait},  Test chorea_acc: {test_acc_chorea}')
        estimators.append(estimator)
    return estimators

def wandb_log(dict_to_log):
    if wandb_flag:
        wandb.log(dict_to_log)
    print(dict_to_log)
def predict_boosting(x,estimators):
    y_pred = estimators[0].predict(x)
    for estimator in estimators[1:]:
        y_pred = update_y_pred(y_pred,estimator.predict(x))
    return y_pred



def calc_gradient(y_pred,y_train):
    y_train_tensor = torch.Tensor(y_train)
    y_pred_tensor = torch.Tensor(y_pred)
    if is_multi_label:
        gradient_tensor = torch.cat((sslmodel.get_gait_grad(y_pred_tensor,y_train_tensor,is_multi_label=True),sslmodel.get_chorea_grad(y_pred_tensor,y_train_tensor,is_multi_label=True)),dim=1)
    else:
        gradient_tensor = sslmodel.get_chorea_grad(y_pred_tensor,y_train_tensor,is_multi_label=False) + sslmodel.get_gait_grad(y_pred_tensor,y_train_tensor,is_multi_label=False)
    gradient_np = gradient_tensor.numpy()
    return gradient_np
    
def update_y_pred(current_pred, new_pred):
    return current_pred + learning_rate*new_pred

def logits_to_predicted_proba(y_pred):
    exp_logits = np.exp(y_pred - np.max(y_pred))
    probabilities  = exp_logits / np.sum(exp_logits, axis=1, keepdims=True) 
    return probabilities

def log_softmax(y_pred):
    max_vals = np.max(y_pred, axis=1, keepdims=True)
    shifted_preds = y_pred - max_vals
    exp_preds = np.exp(shifted_preds)
    row_sums = np.sum(exp_preds, axis=1, keepdims=True)
    softmax_probs = exp_preds / row_sums
    return softmax_probs

def get_valid_chorea(y_train):
    return np.sum(y_train[:,:10], axis=1, keepdims=True)

def get_scores_for_gait_detection(y_true,y_pred):
    pred_y_gait = sslmodel.get_gait(y_pred,is_pred=True)   
    gait_labels = sslmodel.get_gait(y_true,is_pred=False)
    gait_predictions = np.argmax(pred_y_gait,axis=1)
    gait_predictions_logits = pred_y_gait
    return gait_predictions,gait_predictions_logits,gait_labels

def get_scores_for_chorea_detection(y_true,y_pred):
    pred_y_chorea = sslmodel.get_chorea(y_pred, is_pred=True)   
    chorea_labels = sslmodel.get_chorea(y_true, is_pred=False)
    chorea_predictions = np.argmax(pred_y_chorea,axis=1)
    chorea_predictions_logits = pred_y_chorea
    return chorea_predictions,chorea_predictions_logits,chorea_labels


def main():
    if TRAIN_MODE:
        weights_path = os.path.join(OUTPUT_DIR,f'multiclass_weights_hd_only_boosting_shifted_win_std_bandpass_10sec_walk_th.pt')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_class = 10
        input_file = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/data_ready/windows_input_to_multiclass_model_hd_only_10sec_all_shifted_win_std_bandpass_walk_th_corrected_labels.npz')
        win_acc_data = input_file['arr_0']
        win_acc_data = np.transpose(win_acc_data,[0,2,1])
        win_labels = input_file['gait_label_chorea_comb']
        win_subjects = groups =input_file['arr_2']
        win_shift = input_file['win_shift_all_sub']
        
        one_hot_labels = np.zeros((len(win_labels), num_class+2), dtype=int) # adding to extra dummies for no chorea
        one_hot_labels[np.arange(len(win_labels)), win_labels.squeeze().astype(int)] = 1
        X_train = win_acc_data
        y_train = one_hot_labels
        fold = 0
        gait_predictions_all_folds = []
        gait_predictions_logits_all_folds = []
        gait_labels_all_folds = []
        chorea_predictions_all_folds = []
        chorea_predictions_logits_all_folds = []
        chorea_labels_all_folds = []
        valid_chorea_all_folds = []

        cv_test_idxs_all_folds = []
        cv_train_idxs_all_folds = []
        run_num = 30
        for train_idxs, test_idxs in models_new.groupkfold(groups, n_splits=5): 
            if wandb_flag:
                run = wandb.init(project='hd_gait_detection_with_ssl_boosting',
                        reinit=True,
                        group=f"experiment_{run_num}_boosting_multilabel_cv_n_est_{n_estimators}_lr_{learning_rate}_shifted_win_trn_std_bandpass_10sec_walk_th")
                wandb.config.update({'n_estimators':n_estimators, 'boosting_learning_rate': learning_rate})
                run_name = f'experiment_{run_num}_fold_{fold}'
                wandb.run.name = run_name
            
            test_idxs = (test_idxs[0][np.where(win_shift[test_idxs[0]]==0)[0]],) # remove shift != 0 from test
            X_train = win_acc_data[train_idxs]
            X_test = win_acc_data[test_idxs]
            y_train = one_hot_labels[train_idxs]
            y_test = one_hot_labels[test_idxs]
            
            '''
            ## duplicate samples of chorea levels 2-4
            ind_high = np.where(np.sum(y_train[:,4:10],axis=1)>0)[0]
            num_rep = 5
            add_noise = True
            for i in ind_high:
                y_train = np.concatenate([y_train,np.repeat(y_train[i:i+1,:],num_rep,axis=0)],axis=0)
                repeat = np.repeat(X_train[i:i+1,:],num_rep,axis=0)
                if add_noise:
                    win = X_train[i:i+1,:]
                    std = np.std(win, axis=1, keepdims=True)
                    noise = np.random.randn(*repeat.shape) * std * 0.1
                else:
                    noise = 0
                X_train = np.concatenate([X_train,repeat + noise],axis=0)
            permutation_indices = np.random.permutation(len(X_train))
            X_train = X_train[permutation_indices]
            y_train = y_train[permutation_indices]
            '''

           
            cv_test_idxs = test_idxs
            cv_train_idxs = train_idxs

            estimators = train_multiclass(X_train, y_train, X_test, y_test, batch_size=64, device=device, weights_path=weights_path)
            
            y_test_pred = predict_boosting(X_test,estimators)
            gait_predictions,gait_predictions_logits,gait_labels = get_scores_for_gait_detection(torch.Tensor(y_test), torch.Tensor(y_test_pred))
            chorea_predictions,chorea_predictions_logits,chorea_labels = get_scores_for_chorea_detection(torch.Tensor(y_test), torch.Tensor(y_test_pred))
            valid_chorea = get_valid_chorea(y_test)
            test_acc_gait, test_acc_chorea = sslmodel.calc_gait_and_chorea_acc(torch.Tensor(y_test), torch.Tensor(y_test_pred))
            fold_index = len(gait_predictions_all_folds)
            generate_confusion_matrix_per_chorea_lvl(gait_predictions, gait_labels, chorea_predictions, chorea_labels, valid_chorea, fold_index)
            
            gait_predictions_all_folds.append(gait_predictions)
            gait_predictions_logits_all_folds.append(gait_predictions_logits)
            gait_labels_all_folds.append(gait_labels)

            chorea_predictions_all_folds.append(chorea_predictions)
            chorea_predictions_logits_all_folds.append(chorea_predictions_logits)
            chorea_labels_all_folds.append(chorea_labels)
            valid_chorea_all_folds.append(valid_chorea)

            cv_test_idxs_all_folds.append(cv_test_idxs)
            cv_train_idxs_all_folds.append(cv_train_idxs)
            print(f'val_gait_acc_in_fold_{fold}: {test_acc_gait},  chorea_acc_in_fold_{fold}: {test_acc_chorea}')
            wandb_log({'gait_test_acc': test_acc_gait, 'chorea_test_acc': test_acc_chorea})
            if wandb_flag:
                run.finish()
            
            fold+=1
        
        gait_predictions_all_folds = torch.cat(gait_predictions_all_folds)
        gait_predictions_logits_all_folds = torch.cat(gait_predictions_logits_all_folds)  
        gait_labels_all_folds = torch.cat(gait_labels_all_folds)

        chorea_predictions_all_folds = torch.cat(chorea_predictions_all_folds)
        chorea_predictions_logits_all_folds = torch.cat(chorea_predictions_logits_all_folds)
        chorea_labels_all_folds = torch.cat(chorea_labels_all_folds)
        valid_chorea_all_folds = np.concatenate(valid_chorea_all_folds)

        np.savez(os.path.join(OUTPUT_DIR, f'multiclass_separated_labels_predictions_and_logits_with_true_labels_and_subjects_hd_only_boosting_shifted_win_trn_std_bandpass_10sec_walk_th_corrected_labels.npz'),
                    gait_predictions_all_folds=gait_predictions_all_folds,
                    gait_predictions_logits_all_folds=gait_predictions_logits_all_folds,
                    gait_labels_all_folds=gait_labels_all_folds,
                    chorea_predictions_all_folds=chorea_predictions_all_folds,
                    chorea_predictions_logits_all_folds=chorea_predictions_logits_all_folds,
                    chorea_labels_all_folds=chorea_labels_all_folds,
                    valid_chorea_all_folds=valid_chorea_all_folds,
                    win_subjects=win_subjects, 
                    cv_test_idxs_all_folds=cv_test_idxs_all_folds)
        generate_confusion_matrix_per_chorea_lvl(gait_predictions_all_folds, 
                                                gait_labels_all_folds, 
                                                chorea_predictions_all_folds, 
                                                chorea_labels_all_folds, 
                                                valid_chorea_all_folds, 
                                                fold_index='all')
    
    if EVAL_MODE:
        output_file = np.load(os.path.join(OUTPUT_DIR, f'multiclass_separated_labels_predictions_and_logits_with_true_labels_and_subjects_hd_only_boosting_shifted_win_trn_std_bandpass_10sec_walk_th.npz'),allow_pickle=True)
        gait_predictions_all_folds = output_file['gait_predictions_all_folds'],
        gait_predictions_logits_all_folds = output_file['gait_predictions_logits_all_folds'],
        gait_labels_all_folds = output_file['gait_labels_all_folds'],
        chorea_predictions_all_folds = output_file['chorea_predictions_all_folds'],
        chorea_predictions_logits_all_folds = output_file['chorea_predictions_logits_all_folds'],
        chorea_labels_all_folds = output_file['chorea_labels_all_folds'],
        valid_chorea_all_folds = output_file['valid_chorea_all_folds'],
        win_subjects = output_file['win_subjects'], 
        cv_test_idxs_all_folds = output_file['cv_test_idxs_all_folds']
        # debug start
        cv_test_idxs_all_folds_flat = np.concatenate([cv_test_idxs_all_folds[i][0] for i in range(len(cv_test_idxs_all_folds))])
        input_file = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/data_ready/windows_input_to_multiclass_model_hd_only_10sec_all_shifted_win_std_bandpass_walk_th_corrected_labels.npz')
        win_acc_data = input_file['arr_0']
        win_acc_data = np.transpose(win_acc_data,[0,2,1])
        win_video_time = input_file['win_video_time_all_sub']
        
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
        generate_confusion_matrix_per_chorea_lvl(gait_predictions_all_folds[0], 
                                                torch.Tensor(gait_labels_all_folds[0]), 
                                                chorea_predictions_all_folds[0], 
                                                torch.Tensor(chorea_labels_all_folds[0]), 
                                                valid_chorea_all_folds[0], 
                                                fold_index='all')

def generate_confusion_matrix_per_chorea_lvl(gait_predictions, gait_labels, chorea_predictions, chorea_labels, valid_chorea, fold_index):

    gait_labels_ind = torch.argmax(gait_labels, dim=-1)
    chorea_labels_ind = torch.argmax(chorea_labels, dim=-1)
    for is_valid in [0, 1]:
        valid_ind = np.where(valid_chorea == is_valid)[0]
        if is_valid:
            for chorea_level in np.unique(chorea_labels_ind):
                indices = np.where((chorea_labels_ind==chorea_level).flatten() * (valid_chorea == is_valid).flatten())[0]
                gait_predictions_sel = gait_predictions[indices]
                gait_labels_sel = gait_labels_ind[indices]
                confusion_matrix(gait_labels_sel, gait_predictions_sel, prefix1=f'{fold_index}', prefix2=f'{chorea_level}')
        else:
            gait_predictions_sel = gait_predictions[valid_ind]
            gait_labels_sel = gait_labels_ind[valid_ind]
            confusion_matrix(gait_labels_sel, gait_predictions_sel, prefix1=f'{fold_index}', prefix2=f'no_valid_chorea')



def confusion_matrix(labels, predictions, prefix1='', prefix2=''):
    cm = metrics.confusion_matrix(labels, predictions) 
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
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(VIZUALIZE_DIR,f'confusion_matrix_{prefix1}_{prefix2}.png'))
    plt.close('all')
    
def add_noise_to_window(window, noise_std):
    noise = np.random.randn(*window.shape) * noise_std
    return window + noise   
    # wandb_log({"gait_test_acc-std": np.std(all_acc_gait_test)})
    # wandb_log({"avg gait_test_acc": np.mean(all_acc_gait_test)})
    # wandb_log({"chorea_test_acc-std": np.std(all_acc_chorea_test)})
    # wandb_log({"avg chorea_test_acc": np.mean(all_acc_chorea_test)})

    
    #gbgta = train_multiclass(X_train, y_train, batch_size=64, device=device, weights_path=weights_path)


    
    
    
    #get_init_pred = 


if __name__ == '__main__':
    main()

# TODO: 
# 1) write validate funtion: can reference the funcction _validate_model. Note you should use the .predict function
# 2) write a simple main that generate the test data and validation data for one fold 
#      and train the data and print the validation results 
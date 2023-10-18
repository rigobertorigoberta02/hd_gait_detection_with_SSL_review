import preprocessing
import numpy as np
import os
from models_new import GaitDetectorSSL,GaitChoreaDetectorSSL
import torch
from sklearn import metrics
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt
import ipdb
import wandb
import argparse

parser = argparse.ArgumentParser(description='hd_gait_detection_with_SSL')

parser.add_argument(
    '--cohort',
    type=str,
    default='hd',
    help='hd-huntington patients only/hc-healthy control only/mixed')

parser.add_argument(
    '--create-multi-class',
    action='store_true',
    default=True,
    help='if true multiclass of gait and chorea classification is applied')

parser.add_argument(
    '--preprocess-mode',
    action='store_true',
    default=False,
    help='if true the raw data is preprocessed')

parser.add_argument(
    '--cross-val-mode',
    action='store_true',
    default=False,
    help='if true training and validation of the model is applied')

parser.add_argument(
    '--eval-mode',
    action='store_true',
    default=False,
    help='if true evaluation of training results is applied')

parser.add_argument(
    '--gait-all-mode',
    action='store_true',
    default=False,
    help='if true include gait/non-gait window without chorea labels')

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



args = parser.parse_args()




VISUALIZE_ACC_VS_PRED_WIN = False
RAW_DATA_AND_LABELS_DIR = '/home/dafnas1/datasets/hd_dataset/lab_geneactive/synced_labeled_data_walking_non_walking'
PROCESSED_DATA_DIR ='/home/dafnas1/my_repo/hd_gait_detection_with_SSL/data_ready'
OUTPUT_DIR = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs'
VIZUALIZE_DIR = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/multiclass_hd_only/multiclass_separated_labels'

SRC_SAMPLE_RATE = int(100) #hz
STD_THRESH = 0.015
WINDOW_SIZE = int(30*10)
WINDOW_OVERLAP = 0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wandb_flag:
        wandb.init(project='hd_gait_detection_with_ssl')
    if args.preprocess_mode:
        #iterate over subjects and preprocess the data
        win_data_all_sub = np.empty((0,3,WINDOW_SIZE)) 
        win_labels_all_sub = win_subjects = all_subjects = win_chorea_all_sub = win_shift_all_sub = np.empty((0,1))
        StdIndex_all = inclusion_idx = original_data_len = np.empty((0,))
        NumWin = []

        for file in os.listdir(RAW_DATA_AND_LABELS_DIR):
            if args.cohort == 'hc':
                if 'TCCO' in file:
                    data_file = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file))
                else:
                    continue
            if args.cohort == 'hd':
                if 'TCCO' in file:
                    continue
                else:
                    data_file = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file))
            acc_data = data_file['arr_0'].astype('float')
            labels = data_file['arr_1']
            chorea = data_file['arr_2']
            video_time = data_file['arr_3']
            subject_name = file.split('.')[0]
            subject_id = np.tile(subject_name,len(labels)).reshape(-1, 1)
            all_subjects = np.append(all_subjects,subject_id,axis=0)


            ## apply moving standard deviation 
            #data_std = preprocessing.movingstd(data=acc_data,window_size=3*SRC_SAMPLE_RATE)
            data_std = preprocessing.movingstd(data=acc_data,window_size=WINDOW_SIZE)
            StdIndex = data_std > STD_THRESH
            # Remove low activity data (i.e. low std)
            acc_data = acc_data[StdIndex, :]
            labels = labels[StdIndex]
            chorea = chorea[StdIndex]
            video_time = video_time[StdIndex]
            
            ## apply bandpassfilter
            #acc_data = preprocessing.bandpass_filter(data=acc_data,low_cut=0.2,high_cut=15,sampling_rate=SRC_SAMPLE_RATE,order=4)
            acc_data = preprocessing.lowpass_filter(data=acc_data,low_cut=5 ,sampling_rate=SRC_SAMPLE_RATE,order=4)
            ## apply resampling 
            acc_data,labels, chorea, video_time = preprocessing.resample(data=acc_data,labels=labels,chorea=chorea, video_time=video_time ,original_fs=SRC_SAMPLE_RATE,target_fs=30)


            ## deivide data and labels to fixed windows
            data, labels, chorea, video_time, shift, inclusion, NumWinSub = preprocessing.data_windowing(data=acc_data, labels=labels, chorea=chorea, video_time=video_time, window_size = WINDOW_SIZE, window_overlap=WINDOW_OVERLAP,
                                                                                std_idx=StdIndex)
            # Concat the data and labels of the different subjects
            win_data_all_sub = np.append(win_data_all_sub, data, axis=0)
            win_labels_all_sub = np.append(win_labels_all_sub, labels, axis=0)
            win_chorea_all_sub = np.append(win_chorea_all_sub, chorea, axis=0)
            win_shift_all_sub = np.append(win_shift_all_sub, shift, axis=0)

            # Create subject vector that will use for group the data in the training
            subject = np.tile(subject_name, (len(labels), 1)).reshape(-1, 1)
            win_subjects = np.append(win_subjects, subject)

            StdIndex_all = np.append(StdIndex_all, StdIndex, axis=0)
            inclusion_idx = np.append(inclusion_idx.squeeze(), inclusion, axis=0)
            original_data_len = np.append(original_data_len, len(StdIndex))
            NumWin.append(NumWinSub)

        ## Save arrays after preprocessing and windowing
            '''
             with open(os.path.join(args.output_path, "SubjectsVec.p"), 'wb') as outputFile:
            pickle.dump(all_subjects, outputFile)
            # Save the low activity indexes that was filtered out, this will use for the final validation of the model
            with open(os.path.join(args.output_path, "StdIndex.p"), 'wb') as outputFile:
                pickle.dump(StdIndex_all, outputFile)
            # Save the inclusion indices
            with open(os.path.join(args.output_path, "InclusionIndex.p"), 'wb') as outputFile:
                pickle.dump(inclusion_idx, outputFile)
            # Save the number of windows per subject
            with open(os.path.join(args.output_path, "NumWinSub.p"), 'wb') as outputFile:
                pickle.dump(NumWinSub, outputFile)
            '''
        # Save the data, labels and groups
        res = {'win_data_all_sub': win_data_all_sub,
               'win_labels_all_sub': win_labels_all_sub,
               'win_subjects': win_subjects,
               'StdIndex_all': StdIndex_all,
               'original_data_len' : original_data_len,
               'win_chorea_all_sub': win_chorea_all_sub,
               'win_shift_all_sub': win_shift_all_sub,
               }
        # np.savez(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_models_{COHORT}_only.npz'), **res)
        ipdb.set_trace()
        np.savez(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_models_{args.cohort}_only_{args.run_suffix}.npz'), win_data_all_sub, win_labels_all_sub, win_subjects, StdIndex_all, original_data_len, win_chorea_all_sub)
        if args.create_multi_class:
            res = preprocessing.get_label_chorea_comb(res)
            res['arr_0'] = res['win_data_all_sub']
            res['arr_1'] = res['win_labels_all_sub']
            res['arr_2'] = res['win_subjects']
            res['arr_3'] = res['StdIndex_all']
            res['arr_4'] = res['original_data_len']
            res['arr_5'] = res['win_chorea_all_sub']
            res['arr_6'] = res['win_shift_all_sub']
            np.savez(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model_{args.cohort}_only_{args.run_suffix}.npz'), **res)

    if args.cross_val_mode:
        if args.create_multi_class:
            inputs_file = np.load(os.path.join(PROCESSED_DATA_DIR, f'windows_input_to_multiclass_model_{args.cohort}_only_{args.run_suffix}.npz'))
            gait_label_chorea_comb = inputs_file['gait_label_chorea_comb']
            win_labels = gait_label_chorea_comb
            num_class = 10

            ##comparing the multiclass configuration for 2 classes
            #num_class = 2
            #gait_label_chorea_comb = inputs_file['win_labels_all_sub']
            #win_labels = gait_label_chorea_comb
        else:
            inputs_file = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/data_ready/windows_input_to_models_{args.cohort}_only.npz')
            win_labels = inputs_file['arr_1']
            num_class = 2

        win_acc_data = inputs_file['arr_0']
        win_acc_data = np.transpose(win_acc_data,[0,2,1])
        win_subjects = inputs_file['arr_2']
        one_hot_labels = np.zeros((len(win_labels), num_class+2), dtype=int) # adding to extra dummies for no chorea
        one_hot_labels[np.arange(len(win_labels)), win_labels.squeeze().astype(int)] = 1
        
        params = {'learning_rate': 0.0001,
                    'batch_size': 64,
                    'num_epoch': 30,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        
        if args.wandb_flag:
            wandb.config.update(params)
        
        if args.create_multi_class:
            #creating multiclass detector instance
            
            weights_path = os.path.join(OUTPUT_DIR,f'multiclass_weights_{args.cohort}_only_{args.run_suffix}.pt')
            gcd = GaitChoreaDetectorSSL(weights_path=weights_path, device=device, verbose=True,num_classes=num_class)
            gcd, labels, predictions, predictions_log, cv_test_idxs, predictions_train, predictions_log_train, cv_train_idxs = gcd.cross_val(win_acc_data, one_hot_labels, params=params, groups=win_subjects, return_f1=False,wandb_flag=args.wandb_flag)
            np.savez(os.path.join(OUTPUT_DIR, f'multiclass_predictions_and_logits_with_true_labels_and_subjects_{args.cohort}_only_{args.run_suffix}.npz'),predictions,predictions_log,labels,win_subjects, cv_test_idxs, predictions_train, predictions_log_train, cv_train_idxs)
        else:
            #creating walking detector instance
            
            weights_path = os.path.join(OUTPUT_DIR,f'weights_{args.cohort}_only_{args.run_suffix}.pt')
            gd = GaitDetectorSSL(weights_path=weights_path, device=device, verbose=True)
            gd, labels, predictions, predictions_log = gd.cross_val(win_acc_data, one_hot_labels, params=params, groups=win_subjects, return_f1=False)
            np.savez(os.path.join(OUTPUT_DIR, f'predictions_and_logits_with_true_labels_and_subjects_{args.cohort}_only_{args.run_suffix}.npz'),predictions,predictions_log,labels,win_subjects)

    if args.eval_mode:
        ipdb.set_trace()
        data_file = np.load(f'/home/dafnas1/my_repo/hd_gait_detection_with_SSL/data_ready/windows_input_to_multiclass_model_hd_only_{args.run_suffix}.npz')
        win_acc_data = data_file['arr_0']
        win_chorea = data_file['arr_5']
        inputs_file = np.load(f'/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/multiclass_predictions_and_logits_with_true_labels_and_subjects_hd_only_{args.run_suffix}.npz', allow_pickle=True)
        predictions = inputs_file['arr_0']
        predictions_log = inputs_file['arr_1']
        labels = inputs_file['arr_2']
        win_subjects = inputs_file['arr_3']
        cv_test_idxs = inputs_file['arr_4']
        predictions_train = inputs_file['arr_5']
        predictions_log_train = inputs_file['arr_6']
        cv_train_idxs = inputs_file['arr_7']
        cv_train_idxs2 = np.concatenate([val[0] for val in cv_train_idxs])
        labels_train = labels[cv_train_idxs2]
        ipdb.set_trace()
        results_visualization(win_acc_data, predictions, predictions_log,labels,subject_name='all',win_chorea=win_chorea)
        results_visualization(win_acc_data, predictions_train, predictions_log_train,labels_train,subject_name='all_train',win_chorea=win_chorea)
        

        for subject_name in np.unique(win_subjects):
            indx=np.where(win_subjects==subject_name)[0]
            sub_predictions=predictions[indx]
            sub_predictions_log=predictions_log[indx]
            sub_labels=labels[indx]
            sub_data=win_acc_data[indx]
            results_visualization(sub_data,sub_predictions,sub_predictions_log,sub_labels,subject_name=subject_name)

        

        ##confusion_matrics_per_subjet

    
    if VISUALIZE_ACC_VS_PRED_WIN:
        input_data = np.load(os.path.join(PROCESSED_DATA_DIR, 'windows_input_to_models.npz'))
        win_acc_data = input_data['arr_0']
        pred_data = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/predictions_and_logits_with_true_labels_and_subjects.npz')
        predictions = pred_data['arr_0']
        predictions_log = pred_data['arr_1']
        labels = pred_data['arr_2']
        win_subjects = pred_data['arr_3']
        ipdb.set_trace()

def results_visualization(data, predictions,predictions_log,labels,subject_name='all',win_chorea=None):
        ###confusion matrix 
        if args.create_multi_class:
            if args.gait_all_mode:
                confusion_matrix(labels, predictions, subject_name, prefix='_multi_class')
                
            else:
                confusion_matrix(labels, predictions, subject_name, prefix='_multi_class')
                labels_binary = labels >= 5
                predictions_binary = predictions >= 5
                confusion_matrix(labels_binary, predictions_binary, subject_name, prefix='_multi_class_binary')
        else:
            confusion_matrix(labels, predictions, subject_name) 
            if win_chorea is not None:
                for i in range(-1, 5):
                    indices = np.where(np.logical_and(win_chorea>i-1,win_chorea<=i))[0]
                    if  indices.shape==(0,):
                        continue
                    else: 
                        confusion_matrix(labels[indices], predictions[indices], f'{subject_name}_level_{i}') 
        
            ## roc curve
            predictions_log_norm = softmax(predictions_log,axis=1)
            predictions_log_norm_pos = predictions_log[:, 1]
            fpr, tpr, thresholds = metrics.roc_curve(labels, predictions_log_norm_pos)
            roc_auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(VIZUALIZE_DIR,f'roc_curve_{subject_name}.png'))
            plt.close('all')


            data_pow = np.sqrt(np.mean(np.square(data), axis=1))
            data_pow_flat = data_pow.flatten()
            data_pow_norm = data_pow_flat/np.max(np.abs(data_pow_flat))
            win_size = data_pow.shape[-1]
            logits_rep = np.repeat(predictions_log_norm[:,1], win_size)
            pred_rep = np.repeat(predictions, win_size)
            labels_rep = np.repeat(labels, win_size)
            if subject_name=='all':
                pass
            if len(data_pow_norm)<10000:
                x = range(len(data_pow_norm))
                plt.plot(x, data_pow_norm)
                plt.plot(x, logits_rep)
                plt.plot(x, pred_rep, '-.')
                plt.plot(x, labels_rep, '--')
                plt.legend(['acc_pow', 'logits', 'pred', 'labels'])
                plt.savefig(os.path.join(VIZUALIZE_DIR,f'acc_vs_pred_{subject_name}.png'))
                plt.close('all')

            else:
                ipdb.set_trace()

                num_subplots = len(data_pow_norm) // 10000 + 1  # Calculate the number of subplots needed
                for i in range(num_subplots):
                    start_index = i * 10000
                    end_index = (i + 1) * 10000

                    # Create subplots for each section of the data
                    x = range(start_index, min(end_index, len(data_pow_norm)))
                    plt.plot(x, data_pow_norm[start_index:end_index])
                    plt.plot(x, logits_rep[start_index:end_index])
                    plt.plot(x, pred_rep[start_index:end_index], '-.')
                    plt.plot(x, labels_rep[start_index:end_index], '--')
                    plt.legend(['acc_pow', 'logits', 'pred', 'labels'])
                    plt.savefig(os.path.join(VIZUALIZE_DIR,f'acc_vs_pred_{subject_name}_part_{i+1}.png'))
                    plt.close('all')
                
def confusion_matrix(labels, predictions, subject_name, prefix=''):
    if args.gait_all_mode:
        walk = labels%2
        chorea = np.floor(labels/2)
        labels = chorea + walk*6
        walk = predictions%2
        chorea = np.floor(predictions/2)
        predictions = chorea + walk*6
    cm = metrics.confusion_matrix(labels, predictions) 
    class_labels = ["Non-Walking", "Walking"]
    if args.create_multi_class:
        class_labels =  [f"Non-Walking,{i}" for i in range(6)] + [f"Walking,{i}" for i in range(6)]
        if 'binary' in prefix:
            class_labels = ["Non-Walking", "Walking"]
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
        annot_kws={"fontsize": 12})
    ax.set_xticklabels(class_labels, rotation=45, ha='center',fontsize=12)  # Rotate x-axis labels diagonally
    ax.set_yticklabels(class_labels, rotation=45, va='center', fontsize=12)  # Adjust font size of y-axis labels
    
    

    plt.xlabel("Predicted",{"fontsize":14})
    plt.ylabel("True",{"fontsize":14})
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(VIZUALIZE_DIR,f'confusion_matrix_{subject_name}{prefix}.png'))
    plt.close('all')
            



if __name__ == '__main__':
    main()
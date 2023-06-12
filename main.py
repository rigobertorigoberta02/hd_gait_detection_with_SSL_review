import preprocessing
import numpy as np
import os
from models_new import GaitDetectorSSL
import torch
from sklearn import metrics
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt
import ipdb


PREPROCESS_MODE = False
CROSS_VAL_MODE = False
EVAL_MODE = True
VISUALIZE_ACC_VS_PRED_WIN = False
RAW_DATA_AND_LABELS_DIR = '/home/dafnas1/datasets/hd_dataset/lab_geneactive/synced_labeled_data_walking_non_walking'
PROCESSED_DATA_DIR ='/home/dafnas1/my_repo/hd_gait_detection_with_SSL/data_ready'
OUTPUT_DIR = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs'
SRC_SAMPLE_RATE = int(100) #hz
STD_THRESH = 0.015
WINDOW_SIZE = int(30*10)
WINDOW_OVERLAP = 0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if PREPROCESS_MODE:
        #iterate over subjects and preprocess the data
        win_data_all_sub = np.empty((0,3,300)) 
        win_labels_all_sub = win_subjects = all_subjects = win_chorea_all_sub = np.empty((0,1))
        StdIndex_all = inclusion_idx = original_data_len = np.empty((0,))
        NumWin = []

        for file in os.listdir(RAW_DATA_AND_LABELS_DIR):
            if 'TCCO' in file:
                data_file = np.load(os.path.join(RAW_DATA_AND_LABELS_DIR, file))
            else:
                continue
            acc_data = data_file['arr_0'].astype('float')
            labels = data_file['arr_1']
            chorea = data_file['arr_2']
            subject_name = file.split('.')[0]
            subject_id = np.tile(subject_name,len(labels)).reshape(-1, 1)
            all_subjects = np.append(all_subjects,subject_id,axis=0)


            ## apply moving standard deviation 
            data_std = preprocessing.movingstd(data=acc_data,window_size=3*SRC_SAMPLE_RATE)
            StdIndex = data_std > STD_THRESH
            # Remove low activity data (i.e. low std)
            acc_data = acc_data[StdIndex, :]
            labels = labels[StdIndex]
            chorea = chorea[StdIndex]
            ## apply bandpassfilter
            acc_data = preprocessing.bandpass_filter(data=acc_data,low_cut=0.2,high_cut=15,sampling_rate=SRC_SAMPLE_RATE,order=4)

            ## apply resampling 
            acc_data,labels, chorea = preprocessing.resample(data=acc_data,labels=labels,chorea=chorea,original_fs=SRC_SAMPLE_RATE,target_fs=30)

            ## deivide data and labels to fixed windows
            data, labels, chorea, inclusion, NumWinSub = preprocessing.data_windowing(data=acc_data, labels=labels, chorea=chorea, window_size = WINDOW_SIZE, window_overlap=WINDOW_OVERLAP,
                                                                                std_idx=StdIndex)
            # Concat the data and labels of the different subjects
            win_data_all_sub = np.append(win_data_all_sub, data, axis=0)
            win_labels_all_sub = np.append(win_labels_all_sub, labels, axis=0)
            win_chorea_all_sub = np.append(win_chorea_all_sub, chorea, axis=0)

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
        np.savez(os.path.join(PROCESSED_DATA_DIR, 'windows_input_to_models_hc_only.npz'), win_data_all_sub, win_labels_all_sub, win_subjects, StdIndex_all, original_data_len, win_chorea_all_sub)

    if CROSS_VAL_MODE:
        inputs_file = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/data_ready/windows_input_to_models_hc_only.npz')
        win_acc_data = inputs_file['arr_0']
        win_acc_data = np.transpose(win_acc_data,[0,2,1])
        win_labels = inputs_file['arr_1']
        win_subjects = inputs_file['arr_2']
        ipdb.set_trace()
        one_hot_labels = np.zeros((len(win_labels), 2), dtype=int)
        one_hot_labels[np.arange(len(win_labels)), win_labels.squeeze().astype(int)] = 1
        

        #creating walking detector instance
        gd_params = {'device': 'cuda' if torch.cuda.is_available() else 'cpu','batch_size': 64}
        weights_path = os.path.join(OUTPUT_DIR,'weights_hc_only.pt')
        gd = GaitDetectorSSL(weights_path=weights_path, device=device, verbose=True)
        gd, labels, predictions, predictions_log = gd.cross_val(win_acc_data, one_hot_labels, params=gd_params, groups=win_subjects, return_f1=False)

        np.savez(os.path.join(OUTPUT_DIR, 'predictions_and_logits_with_true_labels_and_subjects_hc_only.npz'),predictions,predictions_log,labels,win_subjects)

    if EVAL_MODE:
        data_file = np.load(os.path.join(PROCESSED_DATA_DIR, 'windows_input_to_models_hc_only.npz'))
        win_acc_data = data_file['arr_0']
        win_chorea = data_file['arr_5']
        inputs_file = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/predictions_and_logits_with_true_labels_and_subjects_hc_only.npz')
        predictions = inputs_file['arr_0']
        predictions_log = inputs_file['arr_1']
        labels = inputs_file['arr_2']
        win_subjects = inputs_file['arr_3']

        results_visualization(win_acc_data, predictions, predictions_log,labels,subject_name='all',win_chorea=win_chorea)

        for subject_name in np.unique(win_subjects):
            indx=np.where(win_subjects==subject_name)[0]
            sub_predictions=predictions[indx]
            sub_predictions_log=predictions_log[indx]
            sub_labels=labels[indx]
            sub_data=win_acc_data[indx]
            results_visualization(sub_data,sub_predictions,sub_predictions_log,sub_labels,subject_name=subject_name)

        

        ##confusion_matrics_per_subjet

        ipdb.set_trace()
    
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
        ipdb.set_trace()
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
        plt.savefig(os.path.join('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/healthy_control',f'roc_curve_{subject_name}.png'))
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
            plt.savefig(os.path.join('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/healthy_control',f'acc_vs_pred_{subject_name}.png'))
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
                plt.savefig(os.path.join('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/healthy_control',f'acc_vs_pred_{subject_name}_part_{i+1}.png'))
                plt.close('all')
                
def confusion_matrix(labels, predictions, subject_name):
    cm = metrics.confusion_matrix(labels, predictions) 
    class_labels = ["Non-Walking", "Walking"]
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/healthy_control',f'confusion_matrix_{subject_name}.png'))
    plt.close('all')
            



if __name__ == '__main__':
    main()
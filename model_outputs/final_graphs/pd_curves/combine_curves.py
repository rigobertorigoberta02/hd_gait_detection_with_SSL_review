import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Load data for model 1
precision_classification_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/precision_classification_with_std_rm_with_fine_tuning.npy')
recall_classification_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/recall_classification_with_std_rm_with_fine_tuning.npy')
fpr_classification_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/fpr_classification_with_std_rm_with_fine_tuning.npy')
tpr_classification_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/tpr_classification_with_std_rm_with_fine_tuning.npy')

precision_classification_no_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/precision_classification_with_std_rm.npy')
recall_classification_no_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/recall_classification_with_std_rm.npy')
fpr_classification_no_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/fpr_classification_with_std_rm.npy')
tpr_classification_no_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/tpr_classification_with_std_rm.npy')

# Load data for model 2
precision_segmentation_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/precision_segmentation_triple_wind_with_std_rm_with_fine_tuning.npy')
recall_segmentation_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/recall_segmentation_triple_wind_with_std_rm_with_fine_tuning.npy')
fpr_segmentation_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/fpr_segmentation_triple_wind_with_std_rm_with_fine_tuning.npy')
tpr_segmentation_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/tpr_segmentation_triple_wind_with_std_rm_with_fine_tuning.npy')

precision_segmentation_no_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/precision_segmentation_triple_wind_with_std_rm.npy')
recall_segmentation_no_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/recall_segmentation_triple_wind_with_std_rm.npy')
fpr_segmentation_no_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/fpr_segmentation_triple_wind_with_std_rm.npy')
tpr_segmentation_no_ft = np.load('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/pd_curves/tpr_segmentation_triple_wind_with_std_rm.npy')

# Calculate AUC for Precision-Recall curves
auc_pr_classification_ft = auc(recall_classification_ft, precision_classification_ft)
auc_pr_classification_no_ft = auc(recall_classification_no_ft, precision_classification_no_ft)
auc_pr_segmentation_ft = auc(recall_segmentation_ft, precision_segmentation_ft)
auc_pr_segmentation_no_ft = auc(recall_segmentation_no_ft, precision_segmentation_no_ft)

# Calculate AUC for ROC curves
auc_roc_classification_ft = auc(fpr_classification_ft, tpr_classification_ft)
auc_roc_classification_no_ft = auc(fpr_classification_no_ft, tpr_classification_no_ft)
auc_roc_segmentation_ft = auc(fpr_segmentation_ft, tpr_segmentation_ft)
auc_roc_segmentation_no_ft = auc(fpr_segmentation_no_ft, tpr_segmentation_no_ft)


# Set font sizes
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.titlesize': 20})
plt.rcParams.update({'axes.labelsize': 18})
plt.rcParams.update({'xtick.labelsize': 14})
plt.rcParams.update({'ytick.labelsize': 14})
plt.rcParams.update({'legend.fontsize': 16})

# Plotting the Precision-Recall Curves
plt.figure(figsize=(12, 9))

# Model 1 with fine-tuning
plt.plot(recall_classification_ft, precision_classification_ft, label=f'Basline classification with FT (AUC = {auc_pr_classification_ft:.2f})', linestyle='--')

# Model 1 without fine-tuning
plt.plot(recall_classification_no_ft, precision_classification_no_ft, label=f'Basline classification without FT (AUC = {auc_pr_classification_no_ft:.2f})', linestyle='--')

# Model 2 with fine-tuning
plt.plot(recall_segmentation_ft, precision_segmentation_ft, label=f'J-Net with FT (AUC = {auc_pr_segmentation_ft:.2f})')

# Model 2 without fine-tuning
plt.plot(recall_segmentation_no_ft, precision_segmentation_no_ft, label=f'J-Net without FT (AUC = {auc_pr_segmentation_no_ft:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('Precision_Recall_Curve.png')


# Plotting the ROC Curves
plt.figure(figsize=(12, 9))

# Model 1 with fine-tuning
plt.plot(fpr_classification_ft, tpr_classification_ft, label=f'Basline classification with FT (AUC = {auc_roc_classification_ft:.2f})', linestyle='--')

# Model 1 without fine-tuning
plt.plot(fpr_classification_no_ft, tpr_classification_no_ft, label=f'Basline classification without FT (AUC = {auc_roc_classification_no_ft:.2f})', linestyle='--')

# Model 2 with fine-tuning
plt.plot(fpr_segmentation_ft, tpr_segmentation_ft, label=f'J-Net with FT (AUC = {auc_roc_segmentation_ft:.2f})')

# Model 2 without fine-tuning
plt.plot(fpr_segmentation_no_ft, tpr_segmentation_no_ft, label=f'J-Net without FT (AUC = {auc_roc_segmentation_no_ft:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('ROC_Curve.png')

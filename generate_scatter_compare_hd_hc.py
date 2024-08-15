import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb
import numpy as np
from scipy import stats
import os

def read_csv_column(file_path, column_index=3):
    df = pd.read_csv(file_path)
    return df.iloc[:, column_index]

def calc_per_subject(subjects, col_seg, col_class):
    list_avg_seg = []
    list_avg_class = []
    list_med_seg = []
    list_med_class = []
    # day = read_csv_column(args.file1, column_index=1)
    for subject in np.unique(subjects):
        indices = np.where(subjects==subject)[0]
        avg_seg = np.mean(col_seg[indices])*24*60/100
        avg_class = np.mean(col_class[indices])*24*60/100
        med_seg = np.median(col_seg[indices])*24*60/100
        med_class = np.median(col_class[indices])*24*60/100
        list_avg_seg.append(avg_seg)
        list_avg_class.append(avg_class)
        list_med_seg.append(med_seg)
        list_med_class.append(med_class)
        print(f'subject: {subject}, avg_seg: {avg_seg}, avg_class:{avg_class}')
        print(f'subject: {subject}, med_seg: {med_seg}, med_class:{med_class}')\
        
    return np.array(list_avg_seg), np.array(list_avg_seg), np.array(list_med_seg), np.array(list_med_class)

def min_max_of_arrays(arr1, arr2, arr3, arr4):
    # Concatenate all arrays into a single array
    all_arrays = np.concatenate((arr1, arr2, arr3, arr4))
    
    # Compute the minimum and maximum values
    min_value = np.min(all_arrays)
    max_value = np.max(all_arrays)
    
    return min_value, max_value

def correlation_plot(x, y, filename):
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)

    slope, intercept = np.polyfit(x, y, 1)
    trend_x = np.linspace(np.min(x), np.max(x), len(x))
    trend_line = slope * trend_x + intercept
    plt.plot(trend_x, trend_line, color='red', linestyle='--')
    text_x = min(x) + (max(x) - min(x)) * 0.8
    text_y = min(trend_line) + (max(trend_line) - min(trend_line)) * 0.2
    plt.text(text_x, text_y, f'r={r_s:.2f}', fontsize=12, color='red')


    plt.scatter(x, y)

    plt.xlabel('Median walking time per day [Minutes]', fontsize=14)
    plt.ylabel('UHDRS-TMS', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(filename)
    plt.close()
def main():
    # Parse arguments
    seg_hc_csv_file_name = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/boosting/daily_living_seg_triple_wind_no_std_per_day_segmentation_hc/walking_precent.csv'
    seg_hd_csv_file_name = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/boosting/daily_living_seg_triple_wind_no_std_per_day_segmentation_hd/walking_precent.csv'
    class_hc_csv_file_name = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/boosting/daily_living_classification_no_std_per_day_classification_hc/walking_precent.csv'
    class_hd_csv_file_name = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/boosting/daily_living_classification_no_std_per_day_classification_hd/walking_precent.csv'
    output_file_name = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/daily_living/compare_methods_HD_and_HC_scatter.png'
    tms_walking_time_corr_file_name = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/daily_living/tms_walking_time_corr.png'
    chorea_box_plot_walking_time_corr_file_name = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs/daily_living/chorea_box_plot_walking_time_corr_sns.png'
    tms = np.array([43,	-1,	30,	60,	44,	63,	62,	12,	32,	31,	34,	55,	10,	10,	33,	41,	68,	45,	42,	52])
    chorea_arm_score = np.array([2,3,2,3,2,3,2,1,2,1,2,2,0,0,2,1,3,3,3,3])
    valid_tms_indices = np.where(tms>0)[0].astype(int)
    # Read the second column of each CSV file
    col_seg_hc = read_csv_column(seg_hc_csv_file_name)
    col_seg_hd = read_csv_column(seg_hd_csv_file_name)
    col_class_hc = read_csv_column(class_hc_csv_file_name)
    col_class_hd = read_csv_column(class_hd_csv_file_name)
    subjects_hc = read_csv_column(seg_hc_csv_file_name, column_index=0)
    subjects_hd = read_csv_column(seg_hd_csv_file_name, column_index=0)

    list_avg_seg_hc, list_avg_seg_hc, list_med_seg_hc, list_med_class_hc = calc_per_subject(subjects_hc, col_seg_hc, col_class_hc)
    list_avg_seg_hd, list_avg_seg_hd, list_med_seg_hd, list_med_class_hd = calc_per_subject(subjects_hd, col_seg_hd, col_class_hd)
    
    # Generate scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(col_seg_hc, col_class_hc, alpha=0.5, color='blue', label='Walking percentage hc')
    plt.scatter(col_seg_hd, col_class_hd, alpha=0.5, color='red', label='Walking percentage hd')
    min_val, max_val = min_max_of_arrays(col_seg_hc, col_seg_hd, col_class_hc, col_class_hd)
    # Add y=x line
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='y=x')

    # Add labels and title
    plt.xlabel('Segmentation', fontsize=14)
    plt.ylabel('Classification', fontsize=14)
    plt.legend(fontsize=12,loc='upper right')
    plt.grid(True)

    # Increase tick label size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save the plot
    plt.savefig(output_file_name)
    plt.close()

    # scatter correlation between tms and walking time 
    x = list_med_seg_hd[valid_tms_indices]
    y = tms[valid_tms_indices]
    correlation_plot(x, y, tms_walking_time_corr_file_name)
    x = list_med_seg_hd
    y = chorea_arm_score
    correlation_plot(x, y, tms_walking_time_corr_file_name.replace('tms', 'chorea'))

    control_labels = ['HC'] * len(list_med_seg_hc)
    combined_walking_times = list(list_med_seg_hd) + list(list_med_seg_hc)
    combined_labels = [str(score) for score in chorea_arm_score] + control_labels


    # data = {'Chorea Level': chorea_arm_score, 'Walking Time (minutes)': list_med_seg_hd}
    data = {'Chorea Level': combined_labels, 'Walking Time (minutes)': combined_walking_times}
    order = ['HC', '0', '1', '2', '3']

# Create a box plot with individual data points
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Chorea Level', y='Walking Time (minutes)', data=data, order=order)
    sns.stripplot(x='Chorea Level', y='Walking Time (minutes)', data=data, color='black', alpha=0.6, jitter=False, size=8, order=order)

# Add labels and title
    plt.xlabel('Chorea Level', fontsize=14)
    plt.ylabel('Median walking time per day [Minutes]', fontsize=14)

# Display the plot
    plt.grid(True)
    plt.savefig(chorea_box_plot_walking_time_corr_file_name)
    plt.close('all')

    ipdb.set_trace()
# Create a box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Chorea Level', y='Walking Time (minutes)', data=data)

    # Add labels and title
    plt.xlabel('Chorea Level', fontsize=14)
    plt.ylabel('Median walking time per day [Minutes]', fontsize=14)

    # Display the plot
    plt.grid(True)
    plt.savefig(chorea_box_plot_walking_time_corr_file_name)
    plt.close('all')

if __name__ == '__main__':
    main()

import ssl_boosting
import numpy as np
import os
import os
from datetime import datetime, timedelta
import ipdb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import argparse

parser = argparse.ArgumentParser(description="Process two strings.")

# Add arguments
parser.add_argument('--cohort', type=str, required=True, help='must be hc or hd')
parser.add_argument('--dataset_hd', type=str, required=True, help='must be iwear or pace')
parser.add_argument('--model', type=str, required=True, help='must be classification or segmentation')

# Parse the arguments
args = parser.parse_args()

matplotlib = False
plotly = True


def main(COHORT = 'hc', model_type = 'segmentation', dataset_hd = 'pace'):
    # hd or hc
    # as in the data file feom the preprocessing step
    if model_type == 'classification' and COHORT=='hc':
        INP_PREFIX = 'daily_living_classification_full_files_no_std'
        OUT_PREFIX = 'daily_living_classification_no_std' 

    if model_type == 'segmentation' and COHORT=='hc':
        INP_PREFIX = 'daily_living_segmentation_triple_wind_full_files_no_std'
        OUT_PREFIX = 'daily_living_seg_triple_wind_no_std'
    
    if model_type == 'classification' and COHORT=='hd':
        if dataset_hd == 'iwear':
            INP_PREFIX = 'daily_living_classification_full_files_no_std'
            OUT_PREFIX = 'daily_living_classification_no_std'
        elif dataset_hd == 'pace':
            INP_PREFIX = 'classification_daily_pace'
            OUT_PREFIX = 'classification_daily_pace'

    if model_type == 'segmentation' and COHORT=='hd':
        if dataset_hd == 'iwear':
            INP_PREFIX = 'daily_living_segmentation_triple_wind_full_files_no_std'
            OUT_PREFIX = 'daily_living_seg_triple_wind_no_std'
        elif dataset_hd == 'pace':
            INP_PREFIX = 'segmentation_triple_wind_pace'
            OUT_PREFIX = 'segmentation_triple_wind_daily_pace'

    GAIT_ONLY = True
    gait_only_prefix = "_gait_only" if GAIT_ONLY else ""
    OUTPUT_DIR = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/output_files'
    PLOT_OUTPUT_DIR = f'/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/results_visualization/boosting/{OUT_PREFIX}_per_day_{model_type}_{COHORT}'
    if not os.path.exists(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)
    input_file = np.load(f'/mlwell-data2/dafna/daily_living_data_array/data_ready/windows_input_to_multiclass_model_{COHORT}_only_{INP_PREFIX}.npz')
    output_file = np.load(os.path.join(OUTPUT_DIR, f'multiclass_separated_labels_predictions_and_logits_with_true_labels_and_subjects_{COHORT}_only_boosting_{OUT_PREFIX}'+gait_only_prefix+'.npz'),allow_pickle=True)


    # Function to read the output file and create a dictionary with the remaining seconds until the end of the day
    def calculate_seconds_until_end_of_day(output_file_path):
        times_dict = {}
        
        try:
            with open(output_file_path, 'r') as file:
                for line in file:
                    filename, time_str = line.strip().split(': ')
                    
                    # Parse the extracted time
                    time = datetime.strptime(time_str, '%H:%M:%S')
                    
                    # Calculate the number of seconds until the end of the day
                    end_of_day = datetime.combine(time.date(), datetime.max.time())
                    remaining_seconds = (end_of_day - time).seconds
                    
                    # Add to dictionary
                    times_dict[filename.split("_")[0]] = remaining_seconds
                    
        except FileNotFoundError:
            print(f"The file {output_file_path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        return times_dict
    '''
    # Example usage
    output_file_path = 'path/to/your/output_file.txt'
    remaining_seconds_dict = calculate_seconds_until_end_of_day(output_file_path)
    print(remaining_seconds_dict)
    '''

    def plot_acc(in_data, prefix):
        acc_power = np.sqrt(np.mean(in_data**2, axis=0))
        plt.plot(acc_power)
        plt.title(prefix)
        plt.savefig(os.path.join(PLOT_OUTPUT_DIR,f'{prefix}.png'))
        plt.close('all')

    if model_type == 'segmentation':
        win_subjects = input_file['arr_2'][1:-1:2] 
    else:
        win_subjects = input_file['arr_2']
    win_acc_data = input_file['arr_0']
    valid_ind = input_file['win_video_time_all_sub']
    gait_predictions_all_folds = output_file['gait_predictions_all_folds']
    gait_predictions_logits_all_folds = output_file['gait_predictions_logits_all_folds']

    patient_times = {}
    if dataset_hd == 'pace':
        time_until_end = calculate_seconds_until_end_of_day(f'/home/dafnas1/my_repo/hd_gait_detection_with_SSL/{dataset_hd.upper()}_start_times.txt') 
    else:
        time_until_end = calculate_seconds_until_end_of_day(f'/home/dafnas1/my_repo/hd_gait_detection_with_SSL/{COHORT.upper()}_start_times.txt')
    # with open('/home/dafnas1/my_repo/hd_gait_detection_with_SSL/start_times.txt', 'r') as file:
    #     lines = file.readlines()
    walking_precent = open(os.path.join(PLOT_OUTPUT_DIR, 'walking_precent.csv'), 'w')
    walking_precent.write('subject, day, precent 25, precent 50, precent 75\n')
    all_predictions_per_all_sub_for_24h = []
    for subject_name in np.unique(win_subjects):
        print(subject_name)
        sub_indx=np.where(win_subjects==subject_name)[0]
        sub_predictions=gait_predictions_all_folds[sub_indx]
        sub_logits = gait_predictions_logits_all_folds[sub_indx]
        
        sub_valid_ind = valid_ind[sub_indx]
        sub_acc_data = win_acc_data[sub_indx]
        start_wind = time_until_end[subject_name]//10 + 1
        end_wind = start_wind + 24*3600/10
        day=0

        # while end_wind <= sub_valid_ind.max():
        while end_wind < sub_predictions.shape[0]:
            # extract predictions


            # Your existing code to get the required variables...
            # Assuming variables sub_valid_ind, start_wind, end_wind, sub_predictions, sub_logits, sub_acc_data, subject_name, day, and PLOT_OUTPUT_DIR are defined
            # day_indx = np.where(np.logical_and(sub_valid_ind >= start_wind, sub_valid_ind < end_wind))[0]
            day_indx = np.arange(start_wind, end_wind).astype(int)
            day_predictions = sub_predictions[day_indx]
            day_logits = sub_logits[day_indx]
            if model_type == 'segmentation':
                day_logits = np.transpose(day_logits, (0, 2, 1))
            day_exp = np.exp(day_logits-np.max(day_logits, axis=-1, keepdims=True))
            day_prob = day_exp / (np.sum(day_exp, axis=-1, keepdims=True) + 1e-7)
            #day_valid_indx = sub_valid_ind[day_indx]

            if model_type=='segmentation':
                win_size = day_prob.shape[1]
                day_indx_trans = np.transpose(day_indx)
                window_indices = np.expand_dims(np.linspace(0, 1-1/win_size, win_size), 1)
                day_indx = (day_indx_trans + window_indices).transpose().flatten().astype(int)
            else:
                win_size = 1
            day_acc_data = sub_acc_data[day_indx]
            if model_type == 'segmentation':
                prob_to_plot = day_prob[:, :, 1].flatten()
            else:
                prob_to_plot = day_prob[:, 1]
            walking_th_25 = np.round(np.mean(prob_to_plot>0.25)*10000)/100
            walking_th_50 = np.round(np.mean(prob_to_plot>0.50)*10000)/100
            walking_th_75 = np.round(np.mean(prob_to_plot>0.75)*10000)/100
            walking_precent.write(f'{subject_name}, {day}, {walking_th_25}, {walking_th_50}, {walking_th_75}\n')
            prob_th_50 = prob_to_plot>0.50
            len_24 = int(np.floor(prob_th_50.shape[0]/24)*24)
            prob_reshape = np.reshape(prob_th_50[:len_24]*1.0, [24, -1])
            hour_mean_pred = np.mean(prob_reshape, axis=1)
            all_predictions_per_all_sub_for_24h.append(hour_mean_pred)
            if plotly:

                # Normalize the day indices
                time_hours = (day_indx - start_wind) / 360

                # Create the scatter plot for walking probability
                fig = px.scatter(
                    x=time_hours[::10], 
                    y=prob_to_plot[::10], 
                    labels={'x': 'Time from midnight [hours]', 'y': 'Walking probability'},
                    title=f'Logits {subject_name} - Day {day} walking precent [25, 50, 75] [{walking_th_25}, {walking_th_50}, {walking_th_75}]'
                )
                fig.update_traces(marker=dict(opacity=0.1, color='blue'), selector=dict(mode='markers'))
                # Add valid indices as another scatter plot
                valid_indices = np.isin(np.arange(end_wind - start_wind), day_indx - start_wind).astype(int)
                valid_time_hours = np.arange(end_wind - start_wind) / 360

                # Customize the layout
                fig.update_layout(
                    xaxis_title='Time from midnight [hours]',
                    yaxis_title='Probability / Valid Indices',
                    legend=dict(
                        x=0.8, 
                        y=0.95, 
                        bgcolor='rgba(255, 255, 255, 0.5)', 
                        bordercolor='rgba(0, 0, 0, 0.5)'
                    ),
                    font=dict(size=12)
                )

                # Save the figure
                output_file = os.path.join(PLOT_OUTPUT_DIR, f'logits_{model_type}_{subject_name}_day_{day}.html')
                fig.write_html(output_file)
                # ipdb.set_trace()
                # Optionally, show the figure in the notebook or interactive window
                # fig.show('browser')
            if matplotlib:
                plt.figure(figsize=(14, 7))  # Increased figure size

                plt.scatter((day_indx - start_wind) / 360, prob_to_plot, alpha=0.1, color='blue', label='Walking probability')

                # Scatter plot for valid indices

                valid_indices = np.isin(np.arange(end_wind - start_wind), day_indx - start_wind).astype(int)
                # plt.scatter(np.arange(end_wind - start_wind) / 360, valid_indices, alpha=0.4, color='red', label='Pass std threshold')

                # Adding grid, title, and labels
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.title(f'Logits {subject_name} - Day {day}', fontsize=16)
                plt.xlabel("Time from midnight [hours]", fontsize=14)
                plt.ylabel("Probability / Valid Indices", fontsize=14)

                # Improved legend
                plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

                # Optimize layout
                plt.tight_layout()

                # Save the figure
                plt.savefig(os.path.join(PLOT_OUTPUT_DIR, f'logits_{model_type}_{subject_name}_day_{day}.png'), dpi=300)
                plt.close("all")

            # if subject_name in ['IW1TCCO','IW02CF','IW2TCCO']:

            #     num_images = 10
            #     walking_ind = np.where(day_predictions)[0]
            #     np.random.shuffle(walking_ind)
            #     non_walking_ind = np.where(day_predictions==0)[0]
            #     np.random.shuffle(non_walking_ind)
            #     for i in range(num_images):
            #         walking_data = day_acc_data[walking_ind[i]]
            #         try:
            #             non_walking_data = day_acc_data[non_walking_ind[i]]
            #         except:
            #             continue
            #         plot_acc(walking_data, prefix=f'{subject_name}_walking_day_{day}_{i}')
            #         plot_acc(non_walking_data, prefix=f'{subject_name}_non_walking_day_{day}_{i}')
            #         print(subject_name, day, i)
            
            ssl_boosting.confusion_matrix(np.zeros_like(day_predictions.flatten()), 
                                        day_predictions.flatten(), 
                                        prefix1=f'{model_type}_{subject_name}_day_{day}', 
                                        prefix2=f'{COHORT}_per_day',
                                        output_dir=PLOT_OUTPUT_DIR)
            # update windows
            day += 1
            start_wind = end_wind + 1
            end_wind = start_wind + 24*3600/10
            print(subject_name, start_wind, end_wind)
    walking_precent.close()
    np.save(os.path.join(PLOT_OUTPUT_DIR, 'walking_prob_per_hour.npy'), np.array(all_predictions_per_all_sub_for_24h))
    
    plt.boxplot(np.array(all_predictions_per_all_sub_for_24h)*60, notch=True, patch_artist=True)
    plt.ylim([0, 20])
    plt.xlabel('Time [Hours]')
    plt.ylabel('Walking time [Minutes]')
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, 'minutes_walking_prob_per_hour.png'))
    plt.close("all")
    ipdb.set_trace()

    plt.boxplot(np.array(all_predictions_per_all_sub_for_24h)*100, notch=True, patch_artist=True)
    plt.ylim([0, 20])
    plt.xlabel('time [hours]')
    plt.ylabel('walking percentage [%]')
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, 'walking_prob_per_hour.png'))
    plt.close("all")
    ipdb.set_trace()



def parse_args():
    assert args.cohort in ['hc', 'hd'], 'cohort must be hc or hd'
    assert args.model in ['classification', 'segmentation'], 'cohort must be classification or segmentation'

    main(args.cohort, args.model, args.dataset_hd)

if __name__ == '__main__':
    parse_args()
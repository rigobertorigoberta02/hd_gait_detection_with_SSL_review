from matplotlib import pyplot as plt
import numpy as np
import json
import os
import ipdb

plt.rcParams.update({'font.size': 12})

JSON_FILE = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/scores.json'
OUT_PATH = '/home/dafnas1/my_repo/hd_gait_detection_with_SSL/model_outputs/final_graphs'
NUM_CHOREA_CLASS = 5

def main():
    with open(JSON_FILE, "r") as json_file:
        # Load the JSON data into a Python dictionary
        data = json.load(json_file)
    methods_names = ['Classification (no segmentation)', 'Basic segmentation', 
                     'Padded-window segmentation', 'Triple-window segmentation']
    hd_methods = ['classification_test_final_gait_only', 
               'basic_segmentation_final_gait_only',
               'segmentation_without_edges_overlap_final_1_4_24_gait_only_hd',
               'segmentation_triple_wind_no_shift_final_8_4_24_gait_only_hd',]
    hc_methods = ['classification_7_4_24_chorea_0_only_gait_only_hc',
               'basic_segmentation_hc_7_4_24_chorea_0_only_gait_only_hc',
               'segmentation_triple_wind_hc_7_4_24_chorea_0_only_gait_only_hc',
               'segmentation_without_edges_hc_7_4_24_chorea_0_only_gait_only_hc',]
    groups_labels = ['HC', 'Chorea lvl. 0', 'Chorea lvl.1', 'Chorea lvl.2', 'Chorea lvl. 3', 'Chorea lvl. 4']
    res = {name: np.zeros((NUM_CHOREA_CLASS+1, 2)) for name in methods_names}
    for method, name in zip(hc_methods, methods_names):
        res[name][0,:] = _extract_auc_ci(data[method][_get_key(data[method], 0)])
    
    for method, name in zip(hd_methods, methods_names):
        for chorea_lvl in range(NUM_CHOREA_CLASS):
            res[name][chorea_lvl+1,:] = _extract_auc_ci(data[method][_get_key(data[method], chorea_lvl)])

    bar_plot_per_chorea_lvl(methods_names,res,groups_labels,prefix='segmentation_methods')

    methods_names = ['Triple-window: multi-label', 'Triple-window: gait only', 
                     'Padded-window: multi-label', 'Padded-window: gait only']
    gait_only_vs_multi_methods_hd = ['segmentation_triple_wind_no_shift_final_8_4_24_hd',
                            'segmentation_triple_wind_no_shift_final_8_4_24_gait_only_hd',
                            'segmentation_without_edges_overlap_final_1_4_24_gait_only_hd',
                            'segmentation_without_edges_overlap_final_1_4_24_hd',]
    groups_labels = ['Chorea lvl. 0', 'Chorea lvl.1', 'Chorea lvl.2', 'Chorea lvl. 3', 'Chorea lvl. 4']
    res = {name: np.zeros((NUM_CHOREA_CLASS, 2)) for name in methods_names}
    for method, name in zip(gait_only_vs_multi_methods_hd, methods_names):
        for chorea_lvl in range(NUM_CHOREA_CLASS):
            res[name][chorea_lvl,:] = _extract_auc_ci(data[method][_get_key(data[method], chorea_lvl)])

    bar_plot_per_chorea_lvl(methods_names,res,groups_labels,prefix='gait_only_vs_multi_methods_hd')


def bar_plot_per_chorea_lvl(methods_names,res,groups_labels,prefix='segmentation_methods'):
    for name in methods_names:
        plt.plot(res[name][:,0])
        # Plot the bars
    positions = np.arange(len(groups_labels))*2
    width = 0.4  # Width of the bars
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plotting bars for each metric
    labels = []
    ci_list = []
    width_tag = width*1.0
    locs = [-1.5*width_tag, -0.5*width_tag, 0.5*width_tag, 1.5*width_tag]
    for name, loc in zip(methods_names, locs):
        try:
            labels.append(ax.bar(positions + loc, res[name][:,0], width, label=name, yerr=res[name][:,1]))
            ci_list.append(res[name][:,1])
        except:
            ipdb.set_trace()
    # Adding labels, title, and legend
    ax.set_ylabel('AUC')
    ax.set_xticks(positions)
    ax.set_xticklabels(groups_labels)
    ax.legend(loc='lower left')
    # Function to add labels on top of the bars
    

    # Adding labels for each metric
    for label, ci in zip(labels, ci_list):
        add_labels(label, ax, ci)

    # Centering x ticklabels
    ax.tick_params(axis='x', pad=10)
    ax.set_ylim(0.0, None)

    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, f'{prefix}.png'))
def add_labels(bars, ax, ci):
        for bar, ci_size in zip(bars, ci):
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 + ci_size*350),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
def _get_key(in_dict, chorea_lvl):
    if f'{chorea_lvl}_per_pixel' in in_dict.keys():
        key = f'{chorea_lvl}_per_pixel'
    else:
        key = f'{chorea_lvl}.0_per_pixel'
    return key
def _extract_auc_ci(in_dict):
    auc = np.round(in_dict["auc"]*100)/100
    ci = in_dict["ci"]
    return np.array([auc, ci])

def graph_example():
   # Generate some example data
# Generate some example data
    np.random.seed(10)
    x = np.arange(5)  # x points
    num_bars = 3      # number of bars per x point
    bar_width = 0.2   # width of each bar
    spacing = 0.05     # spacing between groups of bars

    # Generating example data for multiple bars per x point
    data_means = np.random.normal(loc=5, scale=1, size=(len(x), num_bars))
    data_stds = np.random.uniform(0.5, 1, size=(len(x), num_bars))

    # Calculate confidence intervals
    confidence_intervals = 1.96 * (data_stds / np.sqrt(num_bars))  # Assuming a 95% confidence level

    # Plot the bars
    for i in range(num_bars):
        plt.bar(x + i * (bar_width + spacing), data_means[:, i], yerr=confidence_intervals[:, i], 
                width=bar_width, label=f'Bar {i+1}')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Bar Plot with Confidence Intervals')

    # Add legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig(os.path.join(OUT_PATH, 'example.png'))


if __name__ == '__main__':
    main()

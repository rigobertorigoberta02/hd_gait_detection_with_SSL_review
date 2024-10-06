import numpy as np
from scipy import stats
import ipdb
from scipy.stats import chi2_contingency


import scipy.stats as stats

def chi_square_test(women_group1, men_group1, women_group2, men_group2):
    # Construct the observed contingency table
    observed = [
        [women_group1, men_group1],
        [women_group2, men_group2]
    ]
    
    # Calculate row and column totals
    row_totals = [sum(observed[0]), sum(observed[1])]
    col_totals = [observed[0][0] + observed[1][0], observed[0][1] + observed[1][1]]
    grand_total = sum(row_totals)
    
    # Calculate expected frequencies
    expected = [
        [(row_totals[0] * col_totals[0]) / grand_total, (row_totals[0] * col_totals[1]) / grand_total],
        [(row_totals[1] * col_totals[0]) / grand_total, (row_totals[1] * col_totals[1]) / grand_total]
    ]
    
    # Calculate the chi-square statistic
    chi2_stat = sum((observed[i][j] - expected[i][j]) ** 2 / expected[i][j]
                    for i in range(2) for j in range(2))
    
    # Degrees of freedom for a 2x2 table
    df = 1
    
    # Calculate the p-value
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    
    return chi2_stat, p_value

def t_test(group1, group2):
    # Perform the t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    
    return t_stat, p_value

# Usage example
HD_daily_age = [91.17,	60	,64.4	,74.4	,72	,44	,72.2	,82	,99	,86	,69	,82.5	,56	,66	,83.9	,94.3	,61.2	,72.5	,68.8	,56.7]  # Sample data for group 1
HC_daily_age = [68.4	,59.8	,114.4	,97.4	,80.4	,68.8	,107.4	,78.6	,78.8	,96.2	,99.7	,62	,83.9	,79.37	,70.3	,79	,58	,73	,94	,91	,88	,77	,90.7	,82	,66]  # Sample data for group 2

t_stat, p_value = t_test(HD_daily_age, HC_daily_age)

print(f'T-statistic: {t_stat:.3f}')
print(f'p-value: {p_value:.3f}')
# Usage example
women_group1 = 15
men_group1 = 13
women_group2 = 10
men_group2 = 10

chi2_stat, p_value = chi_square_test(women_group1, men_group1, women_group2, men_group2)

print(f'Chi-square statistic: {chi2_stat:.3f}')
print(f'p-value: {p_value:.3f}')

# # Interpret results
# alpha = 0.05
# if p_val < alpha:
#     print("Reject the null hypothesis: There is a significant difference in the proportion of women between the two groups.")
# else:
#     print("Fail to reject the null hypothesis: There is no significant difference in the proportion of women between the two groups.")

# Example data for two groups
HD = np.array([172.72,	170	,167.64	,189,	182.88,	154,	178,	180	,184	,173.7	,184	,169	,182.8	,160	,169	,182.9	,176.7	,170.6	,170	,178	,155.45
])

HC = np.array([166.5,	184,	159.9,	155.4,	170.2,	169.9,	181.8,	185	,171.5,	166	,180	,172.3	,162	,155.5	,172	,155.4	,155.4	,164.6	,180	,163	,160	,180	,165	,180	,183	,185	,180	,170
])

## avereged walking precentage over the days per subject
hd_segmentation_walking_precentage = [2.0816666666666666, 2.045, 3.6483333333333334, 3.16, 3.7233333333333327, 2.8033333333333332, 3.903333333333333, 10.613333333333332, 3.9466666666666668, 4.454999999999999, 7.505000000000002, 0.8783333333333333, 2.6083333333333334, 4.195714285714286, 4.6000000000000005, 6.561666666666667, 2.013333333333333, 5.513333333333333, 2.341428571428571, 2.716666666666667]
hd_classification_walking_precentage = [5.1499999999999995, 7.411666666666666, 10.433333333333332, 4.2075000000000005, 6.63, 5.296666666666667, 6.701666666666665, 13.57, 8.35, 8.743333333333332, 13.301666666666668, 3.1366666666666667, 4.456666666666667, 6.955714285714286, 10.37, 8.746666666666668, 5.974999999999999, 7.386666666666667, 4.555714285714285, 14.218333333333334]
hc_segmentation_walking_precentage = [5.16, 4.274, 2.526, 5.222, 5.563999999999999, 6.434, 5.1000000000000005, 5.161666666666666, 2.1616666666666666, 3.606666666666667, 2.921666666666667, 6.075, 6.908333333333334, 7.248333333333334, 4.333333333333334, 4.798333333333333, 6.353333333333333, 5.801666666666667, 0.22166666666666668, 5.006666666666667, 4.938333333333334, 4.513333333333334, 7.0249999999999995, 2.905, 4.935, 4.113333333333333, 2.3225]
hc_classification_walking_precentage = [0.8666666666666667, 1.186, 1.6280000000000001, 0.632, 2.524, 3.8519999999999994, 0.4483333333333333, 0.4833333333333334, 0.33166666666666667, 0.8200000000000002, 1.6366666666666667, 2.57, 1.6199999999999999, 3.7133333333333334, 2.8699999999999997, 2.0383333333333336, 1.528333333333333, 2.6183333333333336, 0.03333333333333333, 1.4816666666666665, 1.9983333333333337, 0.3183333333333333, 2.1966666666666668, 0.7233333333333333, 0.8300000000000001, 0.8416666666666668, 0.4375]
##  averaged walking mintues over the days per subject
hc_segmentation_walking_mintues = [74.304, 61.54559999999999, 36.374399999999994, 75.1968, 80.12159999999999, 92.64959999999999, 73.44, 74.32799999999999, 31.127999999999997, 51.93600000000001, 42.07200000000001, 87.48, 99.48, 104.376, 62.400000000000006, 69.09599999999999, 91.488, 83.54400000000001, 3.1920000000000006, 72.09599999999999, 71.11200000000002, 64.992, 101.16, 41.832, 71.064, 59.232, 33.443999999999996]
hc_classification_walking_mintues = [12.48, 17.0784, 23.4432, 9.1008, 36.3456, 55.468799999999995, 6.456, 6.960000000000001, 4.776, 11.808000000000002, 23.568, 37.007999999999996, 23.327999999999996, 53.47200000000001, 41.327999999999996, 29.352000000000004, 22.007999999999996, 37.704, 0.48, 21.336, 28.776000000000003, 4.584, 31.631999999999998, 10.415999999999999, 11.952, 12.120000000000003, 6.3]
hd_segmentation_walking_mintues = [29.975999999999996, 29.447999999999997, 52.536, 45.504000000000005, 53.61599999999999, 40.368, 56.20799999999999, 152.832, 56.832, 64.15199999999999, 108.07200000000003, 12.648, 37.56, 60.41828571428572, 66.24, 94.48800000000001, 28.991999999999997, 79.392, 33.71657142857143, 39.12]
hd_classification_walking_mintues = [74.16, 106.728, 150.23999999999998, 60.58800000000001, 95.47200000000001, 76.272, 96.50399999999998, 195.408, 120.23999999999998, 125.90399999999998, 191.544, 45.168, 64.176, 100.16228571428572, 149.328, 125.95200000000001, 86.03999999999998, 106.368, 65.60228571428571, 204.74400000000003]
## median walking minuts over the days per subjct 
hc_segmentation_walking_mintues_median = [74.16000000000001, 66.81599999999999, 33.407999999999994, 89.28, 73.008, 74.304, 77.68799999999999, 64.656, 31.104, 50.904, 42.912, 76.464, 80.63999999999999, 102.024, 55.943999999999996, 50.184, 85.03199999999998, 75.024, 0.0, 78.192, 77.11200000000001, 66.96000000000001, 79.992, 36.647999999999996, 71.85600000000001, 53.92799999999999, 32.904]
hc_classification_walking_mintues_median = [12.096000000000002, 18.864, 21.456, 8.784, 42.336000000000006, 45.36, 6.047999999999999, 7.560000000000001, 3.9600000000000004, 10.368000000000002, 24.336000000000002, 38.304, 19.872, 50.327999999999996, 28.584, 19.152, 21.528000000000002, 29.159999999999997, 0.0, 22.607999999999997, 31.392000000000003, 3.3120000000000003, 25.2, 6.984000000000001, 10.368000000000002, 10.655999999999999, 5.184000000000001]
hd_segmentation_walking_mintues_median = [29.880000000000006, 29.088, 55.008, 46.007999999999996, 50.111999999999995, 41.76, 49.31999999999999, 150.768, 58.751999999999995, 57.672000000000004, 103.24799999999999, 12.527999999999999, 32.472, 65.088, 62.28000000000001, 93.23999999999998, 23.184, 78.04799999999999, 22.896000000000004, 30.6]
hd_classification_walking_mintues_median = [72.71999999999998, 106.92, 146.37599999999998, 62.06400000000001, 101.736, 78.192, 86.4, 190.36800000000002, 117.50399999999999, 121.608, 196.99200000000002, 42.768, 58.248000000000005, 112.32, 149.904, 125.42400000000002, 76.176, 113.83200000000001, 68.4, 197.49599999999998]

tms = np.array([43,	-1,	30,	60,	44,	63,	62,	12,	32,	31,	34,	55,	10,	10,	33,	41,	68,	45,	42,	52])
chorea_arm_score = np.array([2,3,2,3,2,3,2,1,2,1,2,2,0,0,2,1,3,3,3,3])
FA = np.array([20,24,24,21,20,24,9,24,25,20,25,24,25,22,21,21,24,19])
TFC = np.array([10,10,13,10,9,13,5,13,9,12,13,8,14,13,13,8,9,9,12,11])

valid_tms_indices = np.where(tms>0)[0].astype(int)
valid_FA_indices = np.where(FA>0)[0].astype(int)
vaild_TFC_indices = np.where(TFC>0)[0].astype(int)

# ## correlation walking and tms
# ## average - hd - pearson
# print('tms_average_pearson')
# r, p = stats.pearsonr(np.array(hd_classification_walking_mintues)[valid_tms_indices], tms[valid_tms_indices])
# print(f'pearsonr_classification_correlation P:{p} R:{r}')
# r, p = stats.pearsonr(np.array(hd_segmentation_walking_mintues)[valid_tms_indices], tms[valid_tms_indices])
# print(f'pearsonr_segmentation_correlation P:{p} R:{r}')
# r, p = stats.pearsonr(np.array(hd_classification_walking_mintues)[valid_tms_indices]-np.array(hd_segmentation_walking_mintues)[valid_tms_indices], tms[valid_tms_indices])
# print(f'pearsonr_diff_class_seg_correlation P:{p} R:{r}')
# print('tms_average_spearman')
# ## average - hd - spearman
# r, p = stats.spearmanr(np.array(hd_classification_walking_mintues)[valid_tms_indices], tms[valid_tms_indices])
# print(f'spearmanr_classification_correlation P:{p} R:{r}')
# r, p = stats.spearmanr(np.array(hd_segmentation_walking_mintues)[valid_tms_indices], tms[valid_tms_indices])
# print(f'spearmanr_segmentation_correlation P:{p} R:{r}')
# r, p = stats.spearmanr(np.array(hd_classification_walking_mintues)[valid_tms_indices]-np.array(hd_segmentation_walking_mintues)[valid_tms_indices], tms[valid_tms_indices])
# print(f'spearmanr_diff_class_seg_correlation P:{p} R:{r}')

# ## correlation walking and chorea
# print('chorea_average_pearson')
# r, p = stats.pearsonr(np.array(hd_classification_walking_mintues), chorea_arm_score)
# print(f'pearsonr_classification_correlation P:{p} R:{r}')
# r, p = stats.pearsonr(np.array(hd_segmentation_walking_mintues), chorea_arm_score)
# print(f'pearsonr_segmentation_correlation P:{p} R:{r}')
# r, p = stats.pearsonr(np.array(hd_classification_walking_mintues)-np.array(hd_segmentation_walking_mintues), chorea_arm_score)
# print(f'pearsonr_diff_class_seg_correlation P:{p} R:{r}')

# print('chorea_average_spearman')
# r, p = stats.spearmanr(np.array(hd_classification_walking_mintues), chorea_arm_score)
# print(f'spearmanr_classification_correlation P:{p} R:{r}')
# r, p = stats.spearmanr(np.array(hd_segmentation_walking_mintues), chorea_arm_score)
# print(f'spearmanr_segmentation_correlation P:{p} R:{r}')
# r, p = stats.spearmanr(np.array(hd_classification_walking_mintues)-np.array(hd_segmentation_walking_mintues), chorea_arm_score)
# print(f'spearmanr_diff_class_seg_correlation P:{p} R:{r}')
print('median')
## median -hd -  pearson
r, p = stats.pearsonr(np.array(hd_classification_walking_mintues_median)[valid_tms_indices], tms[valid_tms_indices])
print(f'pearsonr_classification_correlation P:{p} R:{r}')
r, p = stats.pearsonr(np.array(hd_segmentation_walking_mintues_median)[valid_tms_indices], tms[valid_tms_indices])
print(f'pearsonr_segmentation_correlation P:{p} R:{r}')
r, p = stats.pearsonr(np.array(hd_classification_walking_mintues_median)[valid_tms_indices]-np.array(hd_segmentation_walking_mintues_median)[valid_tms_indices], tms[valid_tms_indices])
print(f'pearsonr_diff_class_seg_correlation P:{p} R:{r}')

## median - hd - spearman - TMS
print('TMS')
r, p = stats.spearmanr(np.array(hd_classification_walking_mintues_median)[valid_tms_indices], tms[valid_tms_indices])
print(f'spearmanr_classification_correlation P:{p} R:{r}')
r, p = stats.spearmanr(np.array(hd_segmentation_walking_mintues_median)[valid_tms_indices], tms[valid_tms_indices])
print(f'spearmanr_segmentation_correlation P:{p} R:{r}')
r, p = stats.spearmanr(np.array(hd_classification_walking_mintues_median)[valid_tms_indices]-np.array(hd_segmentation_walking_mintues_median)[valid_tms_indices], tms[valid_tms_indices])
print(f'spearmanr_diff_class_seg_correlation P:{p} R:{r}')

## median - hd - spearman - TFC
print('TFC')
r, p = stats.spearmanr(np.array(hd_classification_walking_mintues_median)[vaild_TFC_indices], TFC[vaild_TFC_indices])
print(f'spearmanr_classification_correlation P:{p} R:{r}')
r, p = stats.spearmanr(np.array(hd_segmentation_walking_mintues_median)[vaild_TFC_indices], TFC[vaild_TFC_indices])
print(f'spearmanr_segmentation_correlation P:{p} R:{r}')
r, p = stats.spearmanr(np.array(hd_classification_walking_mintues_median)[vaild_TFC_indices]-np.array(hd_segmentation_walking_mintues_median)[vaild_TFC_indices], TFC[vaild_TFC_indices])
print(f'spearmanr_diff_class_seg_correlation P:{p} R:{r}')

## median - hd - spearman - FA
print('FA')
r, p = stats.spearmanr(np.array(hd_classification_walking_mintues_median)[valid_FA_indices], FA[valid_FA_indices])
print(f'spearmanr_classification_correlation P:{p} R:{r}')
r, p = stats.spearmanr(np.array(hd_segmentation_walking_mintues_median)[valid_FA_indices], FA[valid_FA_indices])
print(f'spearmanr_segmentation_correlation P:{p} R:{r}')
r, p = stats.spearmanr(np.array(hd_classification_walking_mintues_median)[valid_FA_indices]-np.array(hd_segmentation_walking_mintues_median)[valid_FA_indices], FA[valid_FA_indices])
print(f'spearmanr_diff_class_seg_correlation P:{p} R:{r}')

print('chorea_median_spearman')
r, p = stats.spearmanr(np.array(hd_classification_walking_mintues_median), chorea_arm_score)
print(f'spearmanr_classification_correlation P:{p} R:{r}')
r, p = stats.spearmanr(np.array(hd_segmentation_walking_mintues_median), chorea_arm_score)
print(f'spearmanr_segmentation_correlation P:{p} R:{r}')
r, p = stats.spearmanr(np.array(hd_classification_walking_mintues_median)-np.array(hd_segmentation_walking_mintues_median), chorea_arm_score)
print(f'spearmanr_diff_class_seg_correlation P:{p} R:{r}')

print('tms and chorea ')
r, p = stats.spearmanr(tms, chorea_arm_score)
print(f'spearmanr_tms_and_chorea_correlation P:{p} R:{r}')

ipdb.set_trace()
HD = np.array(hd_classification)
HC = np.array(hc_classification)
print(len(HD))
print(len(HC))

# Perform independent samples t-test
t_statistic, p_value = stats.ttest_ind(HD, HC)

# Print results
print("t-statistic:", t_statistic)
print("p-value:", p_value)

# Interpret results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the groups.")
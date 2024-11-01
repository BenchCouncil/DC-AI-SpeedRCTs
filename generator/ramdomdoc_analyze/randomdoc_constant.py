from project_path import pro_path

root = pro_path
final_sample_data = root + f'datasets/Original-Recorded-Version/sample_patient_model.csv'
log_data = root + f'datasets/csv_and_pkl/data_0321_7000.csv'
randomdoc_model_input_path = root + 'datasets/randomdoc_model_input/'
sepsis_3h_label = 'nonsepsis_0'  # sepsis_1, nonsepsis_0(final use version)  #sepsis 3h label is 1 or 0

#Note:Select one model at runtime and comment out the other two.


# ---------------Specialized--------------
model_sort = 'sepsis'
model_input_path = root + 'datasets/sepsis_model_input/'
model_path =root +  f'model_save/sepsis_model/'
first_diag_modelname = 'first_diag_dim_35_kford_2_sen_0.8464_spe_0.6029_acc_75.0_auc_80.97.dat'
click_sequence_modelname = 'model_set_acc_0.8_truedoc_seqlen_5.18_predict_seqlen_6.01.pth'
final_diag_modelname = 'final_diag_dim_36_kford_1_sen_0.9_spe_0.6124_acc_80.3_auc_85.51.dat'

first_diagtime_modelname = 'first_diagtime_dim_35_kford_4_mae_0.456.dat'
final_diagtime_modelname = 'final_diagtime_dim_36_kford_1_mae_0.8018.dat'
feature_dimnum_first = 35
feature_dimnum_final = 36
first_or_final = 'final'  # final , first
ramdoc_nextact_file = root + 'datasets/randomdoc_model_input/sepsis_model_ramdom_doctor_nextact_predict.pkl'
ramdoc_diag_file = root + f'datasets/randomdoc_model_input/sepsis_model_ramdom_doctor_{first_or_final}_diag.pkl'
ramdoc_diagtime_file = root + f'datasets/randomdoc_model_input/sepsis_model_ramdom_doctor_{first_or_final}_diagtime.pkl'


# # ---------------Generalized 0h--------------
# model_sort = 'normal_0h'
# model_input_path = root + 'datasets/normal_0h_model_input/'
# model_path =root +  f'model_save/normal_0h_model/'
# first_diag_modelname = 'first_diag_dim_17_kford_3_sen_0.8006_spe_0.5634_acc_70.51_auc_77.36.dat'
# first_nextact_modelname = 'first_nextact_dim_17_kford_2_auc_86.14_acc_70.27.dat'
# final_diag_modelname = 'final_diag_dim_18_kford_4_sen_0.8542_spe_0.6703_acc_78.98_auc_84.24.dat'
#
# first_diagtime_modelname = 'first_diagtime_dim_17_kford_1_mae_0.432.dat'
# final_diagtime_modelname = 'final_diagtime_dim_18_kford_1_mae_0.7312.dat'
# feature_dimnum_first = 17
# feature_dimnum_final = 18
# first_or_final = 'final'  # first final
# ramdoc_nextact_file = root + 'datasets/randomdoc_model_input/normal_0h_model_ramdom_doctor_nextact_predict.pkl'
# ramdoc_diag_file = None
# ramdoc_diagtime_file = None



# # ---------------Generalized 3h--------------
# model_sort = 'normal_3h'
# model_input_path = root + 'datasets/normal_3h_model_input/'
# model_path =root +  f'model_save/normal_3h_model/'
# first_diag_modelname = 'first_diag_dim_17_kford_2_sen_0.8228_spe_0.554_acc_71.46_auc_77.5.dat'
# first_nextact_modelname = 'first_nextact_dim_17_kford_1_auc_87.44_acc_69.13.dat'
# final_diag_modelname = 'final_diag_dim_18_kford_1_sen_0.863_spe_0.6649_acc_79.36_auc_85.12.dat'
#
# first_diagtime_modelname = 'first_diagtime_dim_17_kford_1_mae_0.4239.dat'
# final_diagtime_modelname = 'final_diagtime_dim_18_kford_1_mae_0.7258.dat'
# feature_dimnum_first = 17
# feature_dimnum_final = 18
# first_or_final = 'final'  #final first
# ramdoc_nextact_file = root + 'datasets/randomdoc_model_input/normal_3h_model_ramdom_doctor_nextact_predict.pkl'
# ramdoc_diag_file = None
# ramdoc_diagtime_file = None




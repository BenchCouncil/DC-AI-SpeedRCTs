from simulator.utils.utils_dataloader import *
import sys
from project_path import pro_path
sys.path.append(pro_path)


first_fn_pkl = pro_path + f'datasets/sepsis_model_input/first_data_7000_dim_35.pkl'
first_diag_data = Data(fn=first_fn_pkl, flag='first')
first_fn_pkl_coxphm = pro_path + f'datasets/sepsis_model_input/first_data_7000_dim_35_coxphm.pkl'
first_diag_data_coxphm = Data(fn=first_fn_pkl_coxphm, flag='first')

final_fn_pkl = pro_path + f'datasets/sepsis_model_input/final_data_7000_dim_36.pkl'
final_diag_data = Data(fn=final_fn_pkl, flag='final')
final_fn_pkl_coxphm = pro_path + f'datasets/sepsis_model_input/final_data_7000_dim_36_coxphm.pkl'
final_diag_data_coxphm = Data(fn=final_fn_pkl_coxphm, flag='final')


sepsis_patient_first_dict = {}
sepsis_patient_final_dict = {}


for patid, emb in zip(first_diag_data.patient_id, first_diag_data.embedding):
    if patid not in (list(sepsis_patient_first_dict.keys())):
        sepsis_patient_first_dict[patid] = emb[-17:]
for patid, emb in zip(first_diag_data_coxphm.patient_id, first_diag_data_coxphm.embedding):
    if patid not in (list(sepsis_patient_first_dict.keys())):
        sepsis_patient_first_dict[patid] = emb[-17:]

for patid, emb in zip(final_diag_data.patient_id, final_diag_data.embedding):
    if patid not in (list(sepsis_patient_final_dict.keys())):
        emb = emb[:-1]
        sepsis_patient_final_dict[patid] = emb[-17:]
for patid, emb in zip(final_diag_data_coxphm.patient_id, final_diag_data_coxphm.embedding):
    if patid not in (list(sepsis_patient_final_dict.keys())):
        emb = emb[:-1]
        sepsis_patient_final_dict[patid] = emb[-17:]



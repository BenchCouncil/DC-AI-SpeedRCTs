import sys
from project_path import pro_path
sys.path.append(pro_path)
import pickle


def read_patient_emb(fn):
    loaded_data = []
    with open(fn, 'rb') as file:
        try:
            while True:
                data = pickle.load(file)
                loaded_data.append(data)
        except EOFError:
            pass  # End of file reached
    unique_id_list, embedding_list = zip(*loaded_data)
    return unique_id_list, embedding_list

def read_first_emb(fn):
    loaded_data = []
    with open(fn, 'rb') as file:
        try:
            while True:
                data = pickle.load(file)
                loaded_data.append(data)
        except EOFError:
            pass  # End of file reached
    uuid_list,patientid_list,embedding_list, diag_list, diag_time_list,percent_action_list = zip(*loaded_data)
    return uuid_list,patientid_list,embedding_list, diag_list, diag_time_list,percent_action_list

def read_click_emb(fn):
    loaded_data = []
    with open(fn, 'rb') as file:
        try:
            while True:
                data = pickle.load(file)
                loaded_data.append(data)
        except EOFError:
            pass  # End of file reached
    uuid_list,patientid_list,embedding_list, click_seqlist = zip(*loaded_data)
    return uuid_list,patientid_list,embedding_list, click_seqlist

def read_click_emb_patdiff(fn):
    loaded_data = []
    with open(fn, 'rb') as file:
        try:
            while True:
                data = pickle.load(file)
                loaded_data.append(data)
        except EOFError:
            pass  # End of file reached
    uuid_list, patid_list, seq_list, cat_embedding_list, click_class_list = zip(*loaded_data)
    return uuid_list, patid_list, seq_list, cat_embedding_list, click_class_list

def read_final_emb(fn):
    loaded_data = []
    with open(fn, 'rb') as file:
        try:
            while True:
                data = pickle.load(file)
                loaded_data.append(data)
        except EOFError:
            pass  # End of file reached
    uuid_list,patientid_list,embedding_list, diag_list, diag_time_list = zip(*loaded_data)
    return uuid_list,patientid_list,embedding_list, diag_list, diag_time_list,None

def read_randomdoc_nextact(fn):
    loaded_data = []
    with open(fn, 'rb') as file:
        try:
            while True:
                data = pickle.load(file)
                loaded_data.append(data)
        except EOFError:
            pass  # End of file reached
    uuid_list,randomdoc_nextact_list = zip(*loaded_data)
    last_slash_index = fn.rfind("/")
    fn_name = fn[last_slash_index + 1:]
    if 'sepsis_model' in fn_name:
        return uuid_list,randomdoc_nextact_list
    else:
        return list(uuid_list[0]),list(randomdoc_nextact_list[0])
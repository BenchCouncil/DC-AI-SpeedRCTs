import sys
from project_path import pro_path
sys.path.append(pro_path)
import pickle
import os
from simulator.data_process.click_sequence.doctor_click import get_click_seq
from simulator.specialized.click_sequence.predict_nextact import nextact_topath,nextact_topath_coxphm,read_nextact_pkl
from simulator.data_process.embedding.doctor_embedding import *
from simulator.data_process.embedding.model_embedding import *
from simulator.data_process.reduce_dim.reduce_dim import pca
from simulator.utils.utils_io_pkl import *

def convert_diag(text):
    label_dict = {
        '无脓毒症': 0,
        '低度疑似脓毒症': 1,
        '高度疑似脓毒症': 1,
        '一般脓毒症': 1,
        '严重脓毒症': 1
    }
    matched_value = None

    for key in label_dict:
        if key in text:
            matched_value = label_dict[key]
            break
    return matched_value



def first_embedding(df,dict_patient_embeddings,model_sort,topath,topath_coxphm,feature_numdim,clickseq):
    for ind, row in df.iterrows():
        patinet_id = row['UNIQUE_ID']
        uuid = row['uuid']
        model_emb = model_embedding(row, model_sort)
        doctor_emb = doctor_embeddimg(row)
        first_diag = convert_diag(row['first_diag'])

        diag_time = row['first_diag_time']
        action_label = row['next_act_percent_label']
        model_str = str(row['AI模型预测结果'])

        if 'TREWScore' not in model_str:
            topath_new = topath
        else:
            topath_new = topath_coxphm

        if model_sort == 'sepsis':#保留患者信息
            if np.all(doctor_emb == 0) or first_diag is None:  # 医生信息为空的样本就过滤掉
                continue
            patient_embedding = dict_patient_embeddings.get(patinet_id)
            if patient_embedding is None:
                continue

            cat_embedding = np.concatenate((doctor_emb, model_emb, patient_embedding), axis=0) #np.array(patient_embedding).flatten()
            if len(cat_embedding) != feature_numdim:
                print('Incorrect total number of model features')

            if clickseq:
                doctor_seq_label = get_click_seq(row['doctor_logid'],patinet_id )
                if doctor_seq_label is not None:  # 不能合并到上面的条件
                    data_to_save = (uuid, patinet_id, cat_embedding, doctor_seq_label)
                    with open(topath_new, 'ab') as file:
                        pickle.dump(data_to_save, file)
            else:
                data_to_save = (uuid, patinet_id, cat_embedding, first_diag, diag_time, action_label)
                with open(topath_new, 'ab') as file:
                    pickle.dump(data_to_save, file)
        else:#普适模型 不保留患者信息
            if np.all(doctor_emb == 0) or first_diag is None:  # 医生信息为空的样本就过滤掉
                continue
            cat_embedding = np.concatenate((doctor_emb, model_emb), axis=0)
            if len(cat_embedding) != feature_numdim:
                print('Incorrect total number of model features')
            data_to_save = (uuid, patinet_id, cat_embedding, first_diag, diag_time, action_label)
            with open(topath_new, 'ab') as file:
                pickle.dump(data_to_save, file)

def get_nextact_by_uuid(uuid_list,nextact_list,uuid_list_coxphm,nextact_list_coxphm,uuid):
    if uuid in uuid_list:
        index = uuid_list.index(uuid)
        return nextact_list[index]
    if uuid in uuid_list_coxphm:
        index = uuid_list_coxphm.index(uuid)
        return nextact_list_coxphm[index]
    return 0

def final_embedding(df, dict_patient_embeddings, model_sort, topath,topath_coxphm,feature_numdim):
    uuid_list = []
    nextact_list = []
    uuid_list_coxphm = []
    nextact_list_coxphm = []
    if model_sort == 'sepsis':
        uuid_list, nextact_list = read_nextact_pkl(nextact_topath)
        uuid_list_coxphm, nextact_list_coxphm = read_nextact_pkl(nextact_topath_coxphm)

    for ind, row in df.iterrows():
        patinet_id = row['UNIQUE_ID']
        uuid = row['uuid']
        model_emb = model_embedding(row, model_sort)
        doctor_emb = doctor_embeddimg(row)
      
        final_diag = convert_diag(row['final_diag'])
        diag_time = row['final_diag_time']
        model_str = str(row['AI模型预测结果'])

        if 'TREWScore' not in model_str:
            topath_new = topath
        else:
            topath_new = topath_coxphm

        if np.all(doctor_emb == 0) or final_diag is None:   # 医生信息为空的样本就过滤掉
            continue

        if model_sort == 'sepsis':  # 保留患者信息
            patient_embedding = dict_patient_embeddings.get(patinet_id)
            if patient_embedding is None:
                continue

            predict_action = get_nextact_by_uuid(uuid_list,nextact_list,uuid_list_coxphm,nextact_list_coxphm,uuid)
            cat_embedding = np.concatenate((doctor_emb, model_emb, patient_embedding, [predict_action]), axis=0)
            if len(cat_embedding) != feature_numdim:
                print('Incorrect total number of model features')
            data_to_save = (uuid, patinet_id, cat_embedding, final_diag, diag_time)
            with open(topath_new, 'ab') as file:
                pickle.dump(data_to_save, file)
        else:  # 普适模型 不保留患者信息
            predict_action = [row['predict_nextact']]
            cat_embedding = np.concatenate((doctor_emb, model_emb, predict_action), axis=0)
            if len(cat_embedding) != feature_numdim:
                print('Incorrect total number of model features')
            data_to_save = (uuid, patinet_id, cat_embedding,final_diag, diag_time)
            with open(topath_new, 'ab') as file:
                pickle.dump(data_to_save, file)


root = f'{pro_path}datasets/csv_and_pkl/'
def model_final_input_emb(model_sort,flag,feature_numdim,pca_dim,clickseq):

    to_root = f'{pro_path}datasets/{model_sort}_model_input/'
    if not os.path.exists(to_root):
        os.makedirs(to_root)
    if flag == 'final' and model_sort != 'sepsis':
        df = pd.read_csv(to_root + f'data_0321_7000_{model_sort}_nextact.csv', encoding='gbk')
        df_coxphm = pd.read_csv(to_root + f'data_0321_7000_{model_sort}_nextact_coxphm.csv', encoding='gbk')
        df = pd.concat([df, df_coxphm])
    else:
        df = pd.read_csv(root + f'data_0321_7000.csv', encoding='gbk')

    patient_embedding = root + f'patient_embedding_{flag}.pkl'
    unique_id_list, patient_embedding_list = read_patient_emb(patient_embedding)
    dim_embedding_list = pca(pca_dim, patient_embedding_list)
    dictionary = dict(zip(unique_id_list, dim_embedding_list))

    if clickseq:
        to_fn = to_root + f'{flag}_data_7000_dim_{feature_numdim}_clickseq.pkl'
    else:
        to_fn = to_root + f'{flag}_data_7000_dim_{feature_numdim}.pkl'

    topath_coxphm = to_fn[:-4] + '_coxphm.pkl'
    if os.path.exists(to_fn):
        os.remove(to_fn)
    if os.path.exists(topath_coxphm):
        os.remove(topath_coxphm)

    if flag == 'first':
        first_embedding(df, dictionary, model_sort, to_fn,topath_coxphm,feature_numdim,clickseq)
    else:
        final_embedding(df, dictionary, model_sort, to_fn,topath_coxphm,feature_numdim)


if __name__ == '__main__':
    #Sepsis simulator or normal simulator (sepsis, normal_0h, normal_3h), preliminary or final (first, final), the dimensions of the final simulator are increased by 1 compared to the preliminary simulator because the percentage of next-step features clicked by the doctor is added.

    emb = sys.argv[1]

    # Sepsis Model，preliminary diagnosis or preliminary diagnosis time simulator input
    if emb == 'sepsis_preliminary':
        model_final_input_emb('sepsis','first',35,17,False)  #第三个参数总维度（模型+患者+医生），第四个维度是患者信息降到多少维度 35=pat:17dim+ doc:8dim+ai model:10dim
    elif emb == 'normal_0h_preliminary':
    # Normal Model 0h  preliminary diagnosis or preliminary diagnosis time or next check percent simulator input
        model_final_input_emb('normal_0h','first',17,None,False) #17 = doc:8dim+ai model:9dim
    elif emb == 'normal_3h_preliminary':
    # Normal Model 3h preliminary diagnosis or preliminary diagnosis time or next check percent simulator input
        model_final_input_emb('normal_3h', 'first',17,None,False)  #17 = doc:8dim+ai model:9dim
    elif emb == 'sequence':
    # Sepsis Model doctor click sequence simulator input
        model_final_input_emb('sepsis', 'first', 35, 17,True)  # 第三个参数总维度（模型+患者+医生），第四个维度是患者信息降到多少维度 35=pat:17dim+ doc:8dim+ai model:10dim
    elif emb == 'sepsis_final':
    # Sepsis Model final diagnosis or final diagnosis time simulator input（need sequence predict result）
        model_final_input_emb('sepsis','final',36,17,False)
    elif emb == 'normal_0h_final':
    # Normal Model 0h final diagnosis or final diagnosis time simulator input
        model_final_input_emb('normal_0h','final',18,None,False) # 18 = doc:8dim+ai model:9dim+ nextact:1dim
    elif emb == 'normal_3h_final':
    # Normal Model 3h final diagnosis or final diagnosis time simulator input
        model_final_input_emb('normal_3h','final',18,None,False) #18 = doc:8dim+ai model:9dim + nextact:1dim

    print(f'{emb} model input embedding complete!')
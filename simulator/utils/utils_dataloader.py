from torch.utils.data import Dataset
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from simulator.utils.utils_io_pkl import *

class Data(Dataset):
    def __init__(self, fn, flag):
        click_seq_list = None
        percent_action_list = None
        diag_list = None
        diag_time_list = None
        seq_list = None
        if 'clickseq' in fn:
            uuid_list,patientid_list,embedding_list, click_seq_list = read_click_emb(fn)
            # uuid_list, patientid_list, seq_list, embedding_list, click_seq_list = read_click_emb_patdiff(fn)

        elif flag == 'first':
            uuid_list,patientid_list,embedding_list, diag_list, diag_time_list,percent_action_list = read_first_emb(fn)
        else:
            uuid_list,patientid_list,embedding_list, diag_list, diag_time_list,percent_action_list = read_final_emb(fn)

        embedding_list = np.array(embedding_list)
        embedding_list = np.where(embedding_list is None, 0.0, embedding_list)
        embedding_list = embedding_list.astype(float)
        embedding_list = np.nan_to_num(embedding_list, nan=0.0, posinf=0.0, neginf=0.0)

        self.uuid = uuid_list
        self.patient_id = patientid_list
        self.embedding = embedding_list
        self.diag = diag_list
        self.diag_time = diag_time_list
        self.percent_action = percent_action_list
        self.click_seq = click_seq_list
        self.seq = seq_list


    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, index):
        return self.embedding[index], self.diag[index], self.diag_time[index]

    def get_emb_by_uuid(self,uuid):
        index = self.uuid.index(uuid)
        # 返回对应的 embedding
        return self.embedding[index]

    def get_diag_by_uuid(self,uuid):
        index = self.uuid.index(uuid)
        # 返回对应的 embedding
        return self.diag[index]

    def get_clickseq_by_uuid(self,uuid):
        uuids = list(self.uuid)
        index = uuids.index(uuid)
        # 返回对应的 embedding
        click_seqs = list(self.click_seq)
        return click_seqs[index]
    # 去除一个患者多个医生诊断的情况
    def remove_multi_doctor_diag(self):
        unique_patient_ids = set()
        unique_indices = []

        # 找出唯一的patient_id及其索引
        for i, pid in enumerate(self.patient_id):
            if pid not in unique_patient_ids:
                unique_patient_ids.add(pid)
                unique_indices.append(i)

        self.uuid = [self.uuid[i] for i in unique_indices]
        self.patient_id = [self.patient_id[i] for i in unique_indices]
        self.embedding = [self.embedding[i] for i in unique_indices]
        self.diag = [self.diag[i] for i in unique_indices]
        self.diag_time = [self.diag_time[i] for i in unique_indices]


def feature_scaling(data):
    # 创建MinMaxScaler实例，设置范围为0到10
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 应用缩放器到数据
    scaled_data = scaler.fit_transform(np.array(data.embedding))
    data.embedding = scaled_data
    return data

def to_xgb(train_set):
    embedding = np.array(train_set.embedding)
    embedding = np.where(np.array(embedding) is None, 0.0, embedding)
    embedding = embedding.astype(float)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    diag = np.array(train_set.diag)
    diag_time = np.array(train_set.diag_time)
    percent_action = np.array(train_set.percent_action)
    return embedding, diag, diag_time,percent_action,

def randomdoc_to_xgb(data1):
    embedding = np.array(data1.embedding)
    embedding = np.where(np.array(embedding) is None, 0.0, embedding)
    embedding = embedding.astype(float)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    data1.embedding = embedding
    return data1

def stratify_sampling_diag(data):
    X_train, X_temp, y_train, y_temp = train_test_split(data.embedding, data.diag, test_size=0.3,random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32, stratify=y_temp,random_state=10)
    train_set = copy.deepcopy(data)
    val_set = copy.deepcopy(data)
    test_set = copy.deepcopy(data)

    train_set.embedding = X_train
    train_set.diag = y_train
    val_set.embedding = X_val
    val_set.diag = y_val
    test_set.embedding = X_test
    test_set.diag = y_test

    train_emb, train_diag, _,_ = to_xgb(train_set)
    val_emb, val_diag, _,_ = to_xgb(val_set)
    test_emb, test_diag, _,_ = to_xgb(test_set)
    return train_emb,train_diag, val_emb,val_diag, test_emb,test_diag


def sampling_diag_of_uuid(data):
    X_train, X_temp, y_train, y_temp = train_test_split(data.uuid, data.diag, test_size=0.3,random_state=42)
    X_val, test_uuid, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32, stratify=y_temp,random_state=10)

    X_train1, X_temp1, y_train1, y_temp1 = train_test_split(data.patient_id, data.diag, test_size=0.3,
                                                        random_state=42)
    X_val, test_patient_id, y_val, _ = train_test_split(X_temp1, y_temp1, test_size=0.32, stratify=y_temp1, random_state=10)

    return test_uuid,test_patient_id,y_test

def sampling_diag_of_uuid_val(data):
    X_train, X_temp, y_train, y_temp = train_test_split(data.embedding, data.diag, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32, stratify=y_temp, random_state=10)

    X_train_uuid, X_temp, y_train_uuid, y_temp = train_test_split(data.uuid, data.diag, test_size=0.3, random_state=42)
    X_val_uuid, X_test, y_val_uuid, y_test = train_test_split(X_temp, y_temp, test_size=0.32, stratify=y_temp, random_state=10)

    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    kflod_uuid = np.concatenate((X_train_uuid,X_val_uuid))
    kflod_em = np.concatenate((X_train,X_val))
    kflod_label = np.concatenate((y_train,y_val))
    val_uuid = None
    val_emb = None
    val_diag = None
    for train_index, val_index in skf.split(kflod_em, kflod_label):
        _, val_uuid = kflod_uuid[train_index], kflod_uuid[val_index]
        _, val_emb = kflod_em[train_index], kflod_em[val_index]
        _, val_diag = kflod_label[train_index], kflod_label[val_index]
        break #最终诊断的kfold是1
    return val_uuid,val_emb,val_diag


def stratify_sampling_diagtime(data,flag):
    data = select_data(data,flag)

    X_train, X_temp, y_train, y_temp = train_test_split(data.embedding, data.diag_time, test_size=0.3,random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32,random_state=42)

    train_set = copy.deepcopy(data)
    val_set = copy.deepcopy(data)
    test_set = copy.deepcopy(data)

    train_set.embedding = X_train
    train_set.diag_time = y_train
    val_set.embedding = X_val
    val_set.diag_time = y_val
    test_set.embedding = X_test
    test_set.diag_time = y_test
    train_emb, _, train_diagtime, _ = to_xgb(train_set)
    val_emb, _, val_diagtime, _ = to_xgb(val_set)
    test_emb, _, test_diagtime, _ = to_xgb(test_set)
    return train_emb,train_diagtime, val_emb,val_diagtime, test_emb,test_diagtime

def sampling_diagtime_of_uuid(data,flag):
    data = select_data(data, flag)

    X_train, X_temp, y_train, y_temp = train_test_split(data.uuid, data.diag_time, test_size=0.3, random_state=42)
    X_val, X_uuid, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32, random_state=42)

    X_train, X_temp, y_train, y_temp = train_test_split(data.patient_id, data.diag_time, test_size=0.3, random_state=42)
    X_val, X_patient_id, y_val, _ = train_test_split(X_temp, y_temp, test_size=0.32, random_state=42)
    return X_uuid,X_patient_id,y_test

def select_data(data,flag):
    embedding_list = []
    diag_time_list = []
    uuid_list =[]
    patid_list =[]

    if flag == 'first': #均值是2.5
        max_value = 7.5 #放大3倍
        min_value = 0.5 #均值的20%
    else:#均值是3.8
        max_value = 15.2 #放大4倍
        min_value = 0.76 #均值的20%
    for i,time in enumerate(data.diag_time):
        if time > max_value or time < min_value:
            print('诊断质量可能不高 去掉')
        else:
            uuid_list.append(data.uuid[i])
            patid_list.append(data.patient_id[i])
            embedding_list.append(data.embedding[i])
            diag_time_list.append(data.diag_time[i])
    data.embedding =embedding_list
    data.diag_time = diag_time_list
    data.uuid = uuid_list
    data.patient_id = patid_list
    return data


def stratify_sampling_nextact(data):
    data.uuid = [uuid for uuid, percent_action in zip(data.uuid, data.percent_action) if percent_action != 1]
    data.patient_id = [patient_id for patient_id, percent_action in zip(data.patient_id, data.percent_action) if
                       percent_action != 1]
    data.embedding = [embedding for embedding, percent_action in zip(data.embedding, data.percent_action) if
                      percent_action != 1]
    data.diag = [diag for diag, percent_action in zip(data.diag, data.percent_action) if percent_action != 1]
    data.diag_time = [diag_time for diag_time, percent_action in zip(data.diag_time, data.percent_action) if
                      percent_action != 1]
    data.percent_action = [percent_action for percent_action in data.percent_action if percent_action != 1]

    X_train, X_temp, y_train, y_temp = train_test_split(data.embedding, data.percent_action, test_size=0.3,random_state=42,stratify=data.percent_action)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32,random_state=42,stratify=y_temp)

    train_set = copy.deepcopy(data)
    val_set = copy.deepcopy(data)
    test_set = copy.deepcopy(data)

    train_set.embedding = X_train
    train_set.percent_action = y_train
    val_set.embedding = X_val
    val_set.percent_action = y_val
    test_set.embedding = X_test
    test_set.percent_action = y_test
    train_emb, _, _, train_next_act = to_xgb(train_set)
    val_emb, _, _, val_next_act = to_xgb(val_set)
    test_emb, _, _, test_next_act = to_xgb(test_set)
    return train_emb,train_next_act, val_emb,val_next_act, test_emb,test_next_act

def sampling_nextact_of_uuid(data):
    data.uuid = [uuid for uuid, percent_action in zip(data.uuid, data.percent_action) if percent_action != 1]
    data.patient_id = [patient_id for patient_id, percent_action in zip(data.patient_id, data.percent_action) if
                       percent_action != 1]
    data.embedding = [embedding for embedding, percent_action in zip(data.embedding, data.percent_action) if
                      percent_action != 1]
    data.percent_action = [percent_action for percent_action in data.percent_action if percent_action != 1]
    X_train, X_temp, y_train, y_temp = train_test_split(data.uuid, data.percent_action, test_size=0.3,random_state=42,stratify=data.percent_action)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32,random_state=42,stratify=y_temp)
    
    X_train, X_temp, y_train, y_temp = train_test_split(data.patient_id, data.percent_action, test_size=0.3,random_state=42,stratify=data.percent_action)

    X_val, x_patid, y_val,_ = train_test_split(X_temp, y_temp, test_size=0.32,random_state=42,stratify=y_temp)
 
    return X_test,x_patid,y_test


def data_split(data,flag, model):
    data = feature_scaling(data)
    if model == 'diag':
        train_em, train_label, val_em, val_label, test_em, test_label = stratify_sampling_diag(data)

    elif model == 'diagtime':
        train_em, train_label, val_em, val_label, test_em, test_label = stratify_sampling_diagtime(data,flag)
    else :
        #simulator =='nextact'
        train_em, train_label, val_em, val_label, test_em, test_label = stratify_sampling_nextact(data)
    print(f'totaldata sum {len(train_em)+len(val_em)+len(test_em)}')
    if model != 'clickseq' and model != 'diagtime':
        print(f'train sum {len(train_em)}, positive num {sum(train_label)}, negative num {len(train_label) - sum(train_label)}')
        print(f'val sum {len(val_em)}, positive num {sum(val_label)}, negative num {len(val_label) - sum(val_label)}')
        print(f'test sum {len(test_em)}, positive num {sum(test_label)}, negative num {len(test_label) - sum(test_label)}')
    return np.concatenate((train_em,val_em)), np.concatenate((train_label,val_label)), test_em, test_label


#At that time the year was processed to exceed the maximum year in mimic
#The dict is meant to be able to be converted to datatime, and the original year exceeds the datatime's maximum value
diag_to_datatime_dict= {
    '2500-':'2000-',
    '2501-':'2001-',
    '2502-':'2002-',
    '2503-': '2003-',
    '2504-': '2004-',
    '2505-': '2005-'
}
def convert_datatime(text):
    text_after = None
    for key in diag_to_datatime_dict.keys():
        if key in text:
            text_after = str(text).replace(key,diag_to_datatime_dict.get(key))
            return text_after
    return text_after

def df_convert_datatime(df,rowname):
    for index,row in df.iterrows():
        text = row[rowname]
        for key in diag_to_datatime_dict.keys():
            if key in text:
                text_after = str(text).replace(key,diag_to_datatime_dict.get(key))
                df.at[index,rowname] = text_after
    return df



def coxphm_test_diag(data_test):
    test_em, test_label, _, _ = to_xgb(data_test)
    print(f'totaldata sum {len(test_em)}')
    print(f'test sum {len(test_em)}, positive num {sum(test_label)}, negative num {len(test_label) - sum(test_label)}')
    return test_em, test_label

def coxphm_test_diagtime(data_test,flag):
    data_test = select_data(data_test,flag)
    train_emb, _, train_diagtime, _ = to_xgb(data_test)
    return train_emb,train_diagtime

def coxphm_test_nextact(data_test):
    # 全部查看就2条数据，去掉，改成8分类了
    data_test.uuid = [uuid for uuid, percent_action in zip(data_test.uuid, data_test.percent_action) if
                           percent_action != 1]
    data_test.patient_id = [patient_id for patient_id, percent_action in
                                 zip(data_test.patient_id, data_test.percent_action) if
                                 percent_action != 1]
    data_test.embedding = [embedding for embedding, percent_action in
                                zip(data_test.embedding, data_test.percent_action) if
                                percent_action != 1]
    data_test.percent_action = [percent_action for percent_action in data_test.percent_action if
                                     percent_action != 1]

    test_em, _, _, test_label = to_xgb(data_test)
    print(f'totaldata sum {len(test_em)}')
    print(f'test sum {len(test_em)}, positive num {sum(test_label)}, negative num {len(test_label) - sum(test_label)}')
    return test_em, test_label

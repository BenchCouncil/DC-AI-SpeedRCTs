import sys
from project_path import pro_path
sys.path.append(pro_path)
import numpy as np
import random
from sklearn.model_selection import StratifiedShuffleSplit
from simulator.utils.utils_dataloader import *
import os


def remove_serial_dupl(lst):
    if not lst:
        return []

    result = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            result.append(lst[i])

    return result

def get_batch_indices(total_length, batch_size):
    assert (batch_size <=
            total_length), ('Batch size is large than total data length.'
                            'Check your data or change batch size.')
    current_index = 0
    indexs = [i for i in range(total_length)]
    random.shuffle(indexs)
    while 1:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexs[current_index:current_index + batch_size], current_index


root = pro_path
fn_pkl = root + f'datasets/sepsis_model_input/first_data_7000_dim_35_clickseq.pkl'
fn_pkl_coxphm = root + f'datasets/sepsis_model_input/first_data_7000_dim_35_clickseq_coxphm.pkl'

if os.path.exists(fn_pkl):
    data_temp = Data(fn=fn_pkl,flag = 'first')
else:
    data_temp = None
if os.path.exists(fn_pkl_coxphm):
    data_coxphm = Data(fn=fn_pkl_coxphm,flag = 'first')
else:
    data_coxphm = None

data = copy.deepcopy(data_coxphm)
data_coxphm_after = copy.deepcopy(data_coxphm)
data_temp_after = copy.deepcopy(data_temp)
if data_temp is not None and data_coxphm is not None:
    data.embedding = np.concatenate((data_temp.embedding, data_coxphm.embedding), axis=0)
    data.click_seq = data_temp.click_seq + data_coxphm.click_seq
    data.uuid = data_temp.uuid + data_coxphm.uuid



tgt_vocab = {'P': 0, 'S': 1, 'E': 2,'历史基础信息': 3, '既往病史': 4, '历史_血常规': 5, '历史_动脉血气分析': 6, '历史_止凝血': 7, '历史_影像检查': 8, '历史_病原检查': 9,'历史_培养': 10,
             '历史_涂片': 11,'历史用药': 12,'下一步_降钙素原': 13,'下一步_血常规': 14,'下一步_动脉血气分析': 15,'下一步_止凝血': 16,'下一步_影像检查': 17,'下一步_病原检查': 18,'下一步_培养': 19,'下一步_涂片': 20}
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}   # 把目标字典转换成 索引：字的形式


def train_test_load(resample):
    data.embedding = data.embedding.astype(int)
    arr = data.embedding
    arr_integer = np.floor(np.abs(arr)*10)
    value_to_index = {}
    index = 0
    for i, row in enumerate(arr_integer):
        for j, element in enumerate(row):
            if element not in list(value_to_index.keys()):
                value_to_index[element] = index
                index+=1
    arr_integer_index = np.zeros_like(arr_integer)
    for i in range(arr_integer.shape[0]):
        for j in range(arr_integer.shape[1]):
            arr_integer_index[i, j] = value_to_index[arr_integer[i, j]]
    data.embedding = arr_integer_index

    embedding_after0 = []
    click_seq_after0 = []
    uuid_after0 = []

    for emb, click,uuid in zip(data.embedding, data.click_seq,data.uuid):
        click = remove_serial_dupl(click)
        if len(click) <= 21 and len(click)>=1: #因为分层采样有的不够 所以小于21
        # if len(click) >= 1:  # 因为分层采样有的不够 所以小于21

            embedding_after0.append(emb)
            click_seq_after0.append(click)
            uuid_after0.append(uuid)
    data.embedding = np.array(embedding_after0)
    data.click_seq = click_seq_after0
    data.uuid = uuid_after0


    max_length = max(len(seq) for seq in data.click_seq)
    print(f'max_length {max_length}')
    Y_np = np.zeros((len(data.click_seq), max_length+2), dtype=int)
    for i, seq in enumerate(data.click_seq):
        seq_n = [1] + [int(x) + 3 for x in seq] +[2] #添加开始符、结束符
        Y_np[i, :len(seq_n)] = np.array(seq_n, dtype=int)

    data.click_seq = Y_np

    embedding_after = []
    click_seq_after = []
    for emb,click in zip(data.embedding,data.click_seq):
        seq_len = np.count_nonzero(click)
        if seq_len not in [23,28,29,32,33]:
            embedding_after.append(emb)
            click_seq_after.append(click)
    data.embedding = np.array(embedding_after)
    data.click_seq = np.array(click_seq_after)
    data.uuid = np.array(data.uuid)

    X_train = None
    X_test = None
    Y_train = None
    Y_test = None
    Y_train_uuid = None
    Y_test_uuid = None

    index_train_val_test = []
    index_comphm = []
    for idx, (uuid, emb, click_s) in enumerate(zip(data.uuid, data.embedding, data.click_seq)):
        if uuid in data_temp.uuid:
            index_train_val_test.append(idx)
        else:
            index_comphm.append(idx)
    data_temp_after.embedding, data_coxphm_after.embedding = data.embedding[index_train_val_test], data.embedding[
        index_comphm]
    data_temp_after.click_seq, data_coxphm_after.click_seq = data.click_seq[index_train_val_test], data.click_seq[
        index_comphm]
    data_temp_after.uuid, data_coxphm_after.uuid = data.uuid[index_train_val_test], data.uuid[index_comphm]


    non_zero_counts = np.count_nonzero(data_temp_after.click_seq, axis=1)
    keep_indices = ~np.isin(non_zero_counts, [23,28, 29, 32, 33])
    non_zero_counts_after = non_zero_counts[keep_indices]
    unique_counts, counts = np.unique(non_zero_counts_after, return_counts=True)
    sparse_classes = unique_counts[counts < 2]
    if len(sparse_classes) > 0:
        print("Categories with insufficient sample size：", sparse_classes)
    else:
        print("Sample size is sufficient for all categories")


    sss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss_train_test.split(data_temp_after.embedding, non_zero_counts_after):
        X_train, X_test = data_temp_after.embedding[train_index], data_temp_after.embedding[test_index]
        Y_train, Y_test = data_temp_after.click_seq[train_index], data_temp_after.click_seq[test_index]
        Y_train_uuid, Y_test_uuid = data_temp_after.uuid[train_index], data_temp_after.uuid[test_index]


    # 将训练集数据复制多份并与原始训练集数据合并
    X_train_augmented = np.copy(X_train)
    Y_train_augmented = np.copy(Y_train)
    Y_train_uuid_augmented = np.copy(Y_train_uuid)

    for _ in range(resample - 1):
        X_train_copy = np.copy(X_train)
        Y_train_copy = np.copy(Y_train)
        Y_train_uuid_copy = np.copy(Y_train_uuid)
        X_train_augmented = np.concatenate([X_train_augmented, X_train_copy], axis=0)
        Y_train_augmented = np.concatenate([Y_train_augmented, Y_train_copy], axis=0)
        Y_train_uuid_augmented = np.concatenate([Y_train_uuid_augmented, Y_train_uuid_copy], axis=0)


    shuffle_indices = np.random.permutation(len(X_train_augmented))
    X_train_augmented = X_train_augmented[shuffle_indices]
    Y_train_augmented = Y_train_augmented[shuffle_indices]
    Y_train_uuid_augmented = Y_train_uuid_augmented[shuffle_indices]

    print(f'train num {X_train_augmented.shape[0]}')
    print(f'test num {X_test.shape[0]}')

    return X_train_augmented,Y_train_augmented,Y_train_uuid_augmented,X_test, Y_test,Y_test_uuid,data_coxphm_after.embedding,data_coxphm_after.click_seq,data_coxphm_after.uuid


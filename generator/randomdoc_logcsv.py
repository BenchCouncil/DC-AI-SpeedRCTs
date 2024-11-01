import sys
from project_path import pro_path
sys.path.append(pro_path)
import pandas as pd
import numpy as np
from randomdoc_num import random_doc_list
import os
import csv
import uuid
import re
import ast
import copy
from  get_patemb_from_model_input import sepsis_patient_first_dict,sepsis_patient_final_dict




def model_data(text):
    if str(text) == 'nan':
        return None, None, None, None, None
    if 'TREWScore' in str(text):
        text = re.sub(r":(\w+),", r":'\1',", text, count=2)
    data_dict = ast.literal_eval(text)
    model_sort = list(data_dict.keys())[0]
    model_pre = data_dict[model_sort]
    second_key = list(data_dict.keys())[1]
    model_visible = data_dict[second_key]
    third_key = list(data_dict.keys())[2]
    model_prob = data_dict[third_key]
    index1 = model_prob.find('0h')
    index2 = model_prob.find(',')
    index3 = model_prob.find('3h')
    model_prob_0h = model_prob[index1 + 3:index2]
    model_prob_3h = model_prob[index3 + 3:]
    # print(
    #     f'model_sort {model_sort},model_visible {model_visible}, model_pre{model_pre},model_prob_0h {model_prob_0h},model_prob_3h {model_prob_3h}')
    return model_sort, model_visible, model_pre, model_prob_0h, model_prob_3h



root = f'{pro_path}datasets/Original-Recorded-Version/'
df_sample = pd.read_csv(root +'sample_patient_model.csv', encoding='gbk')
to_file = f'{pro_path}datasets/csv_and_pkl/data_randomdoc.csv'
if not os.path.exists(f'{pro_path}datasets/csv_and_pkl'):
    os.makedirs(f'{pro_path}datasets/csv_and_pkl')


if __name__ == '__main__':
    patid_list = list(set(sepsis_patient_first_dict.keys()) & set(sepsis_patient_final_dict.keys()))
    df_sample = df_sample[df_sample['UNIQUE_ID'].isin(patid_list)]

    df_doctor_group = df_sample.groupby(['医生ID'])
    print(len(df_doctor_group)) #现在是100组
    add_index = 0
    df_result = pd.DataFrame()
    for name, group in df_doctor_group:
        if len(group) < 60:
            num_to_add = 60 - len(group)
            rows_to_add = df_sample.sample(n=num_to_add, replace=True).copy()
            rows_to_add['医生ID'] = name

            group = pd.concat([group, rows_to_add])
            if add_index < 12:
                add_index = add_index + 1
                group_copy = copy.deepcopy(group)
                group_copy['医生ID'] = 1000 + add_index
                group = pd.concat([group, group_copy])
        df_result = pd.concat([df_result, group])

    df_doctor_group = df_result.groupby(['医生ID'])
    doctor_ids = df_doctor_group.size().index.tolist()

    np.random.seed(42)
    selected_ids = np.random.choice(doctor_ids, 125-len(doctor_ids), replace=False)
    df_selected = df_result[df_result['医生ID'].isin(list(selected_ids))]
    df_selected['医生ID'] = df_selected['医生ID'] + 200
    df_extended = pd.concat([df_result, df_selected])
    df_extended_group = df_extended.groupby(['医生ID'])
    print(f"Total groups after extension: {len(df_extended_group)}")

    #random_doc:virdoc_unit, group_sex, group_age, group_year, title, group_field, group_depart
    index = 0


    for docid,df_row in df_extended_group:
        del_list = ['GROUP', '基础信息（当前）中补充的数据', '下一步检查（当前）中补充的数据', '医生ID']
        for del_sub in del_list:
            del df_row[del_sub]
        df_row['model_sort'] = None
        df_row['model_visible'] = None
        df_row['model_pre'] = None
        df_row['model_prob_0h'] = None
        df_row['model_prob_3h'] = None
        df_row['uuid'] = None

        virdoc = random_doc_list[index]
        index = index+1

        
        for i,row in df_row.iterrows():
            model_sort, model_visible, model_pre, model_prob_0h, model_prob_3h = model_data(
                row['AI模型预测结果'])
            df_row.at[i, 'uuid'] = uuid.uuid4()
            df_row.at[i, 'model_sort'] = model_sort
            df_row.at[i, 'model_visible'] = model_visible
            df_row.at[i, 'model_pre'] = model_pre
            df_row.at[i, 'model_prob_0h'] = model_prob_0h
            df_row.at[i, 'model_prob_3h'] = model_prob_3h

        df_row['diag_order'] = range(1, 61)
        df_row['virdoc_unit'] = virdoc[0]
        df_row['virdoc_sex'] = virdoc[1]
        df_row['virdoc_age'] = virdoc[2]
        df_row['virdoc_year'] = virdoc[3]
        df_row['virdoc_title'] = virdoc[4]
        df_row['virdoc_field'] = virdoc[5]
        df_row['virdoc_depart'] = virdoc[6]

        if os.path.exists(to_file):
            df_row.to_csv(to_file, mode='a', index=False, encoding='gbk', header=False, quoting=csv.QUOTE_ALL)
        else:
            df_row.to_csv(to_file, mode='w', index=False, encoding='gbk', header=True, quoting=csv.QUOTE_ALL)

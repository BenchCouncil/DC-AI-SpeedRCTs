import sys
from project_path import pro_path
sys.path.append(pro_path)
import re
import ast
import pandas as pd
import os
import csv
pd.options.mode.chained_assignment = None
import csv
import uuid
from simulator.utils.utils_dataloader import convert_datatime,df_convert_datatime


# 准备虚拟医生模型的训练数据
exec_hist_dict = {
    '1': '下一步检查_血常规（历史）',
    '2': '下一步检查_动脉血气分析（历史）',
    '3': '下一步检查_止凝血（历史）',
    '4': '下一步检查_影像报告（历史）',
    '5': '下一步检查_病原血检查（历史）',
    '6': '下一步检查_培养（历史）',
    '7': '下一步检查_涂片（历史）',
    '8': '历史用药'
}

def final_train_data(df_doctor_diag, df_syslog_group, df_patient_check_group,df_doctor_diag_group, doctor_patient_set,df_sample,df_docinfo,to_file):
    i = 0
    for doc_pat_id in doctor_patient_set:
        doctor_id = doc_pat_id[0]
        patient_id = doc_pat_id[1]
        # ① Patient info and model info #patient ids 1-6000 Another copy was made at that time and the ids added 20,000 overall
        if patient_id > 20000:
            patient_id = patient_id - 20000
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
            patient_id = patient_id + 20000
        else:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]

        model_sort, model_visible, model_pre, model_prob_0h, model_prob_3h = model_data(
            df_sample_id.iloc[0]['AI模型预测结果'])

        # ③ Order of diagnosis by the doctor
        diag_seq = diag_sequence(df_doctor_diag, doc_pat_id)
        # ④ Doctor's diagnosis Doctor's diagnosis time
        df_doctor_diag_id = df_doctor_diag_group.get_group(doc_pat_id)
        # ⑤ Doctor's Information
        df_docinfo_id = df_docinfo[df_docinfo['doctor_logid'] == doctor_id]

        df_sys_log_id = df_syslog_group.get_group(doc_pat_id)
        first_diag, final_diag = get_final_diag(df_doctor_diag_id)

        if final_diag == 'nan' or first_diag == 'nan':
            continue

        final_diag_time = get_diag_time(df_sys_log_id, df_doctor_diag_id)
        first_diag_time = get_first_diag_time(df_sys_log_id, df_doctor_diag_id)
        if final_diag_time < first_diag_time:
            continue

        percent_action = view_next_feature(df_patient_check_group, doctor_id, patient_id)

        df_combine = combine_feat(df_sample_id, df_docinfo_id, model_sort, model_visible, model_pre, model_prob_0h,
                                  model_prob_3h, diag_seq, first_diag, final_diag, first_diag_time, final_diag_time,
                                  percent_action)
        pd.set_option('display.max_columns', None)

        print('==========')
        print(df_combine)
        print(len(df_combine))
        i = i + 1
        print(f'Total number of lines {i}')
        if os.path.exists(to_file):
            df_combine.to_csv(to_file, mode='a', index=False, encoding='gbk', header=False, quoting=csv.QUOTE_ALL)
        else:
            df_combine.to_csv(to_file, mode='w', index=False, encoding='gbk', header=True, quoting=csv.QUOTE_ALL)



def view_next_feature(df_patient_check_group,doctor_id,patient_id):
    #Return to view percentage of next tests, seven in total
    if (doctor_id,patient_id) in df_patient_check_group.groups.keys():
        df_patient_check = df_patient_check_group.get_group((doctor_id,patient_id))
        exam_type = len(set(df_patient_check['exam_type']))
        precent = round(exam_type / 8, 4)
    else:
        precent = 0
    return precent

#①获取医生诊断结果
def get_final_diag(df_doctor_diag_id):
    df_doctor_diag_id = df_doctor_diag_id.sort_values(by='time', ascending=False)
    first_diag = df_doctor_diag_id.iloc[0]['primary_diag']
    final_diag = df_doctor_diag_id.iloc[0]['final_diag']
    return str(first_diag).strip(),str(final_diag).strip()



#②获取医生最终诊断的时间
def get_diag_time(df_sys_log_id,df_doctor_diag_id):
    df_doctor_diag_id = df_doctor_diag_id[df_doctor_diag_id['operation'] != '点击下一个患者时自动保存信息']
    df_diag_last = df_doctor_diag_id.sort_values(by='time',ascending=False)
    endtime = pd.to_datetime(convert_datatime(df_diag_last.iloc[0]['time_text']))
    df_sys_log_id = df_convert_datatime(df_sys_log_id,'create_time')
    df_sys_log_id.loc[:, 'create_time'] = pd.to_datetime(df_sys_log_id['create_time'])
    df_sys_log_id = df_sys_log_id[df_sys_log_id['create_time'] > endtime-pd.Timedelta(hours=1)]
    df_sys_log_id = df_sys_log_id[df_sys_log_id['module'] != '注销登录']

    df_diag_last = df_sys_log_id.sort_values(by='create_time', ascending=True)
    starttime = df_diag_last.iloc[0]['create_time']

    #按照分钟算
    time_diff = (int(endtime.timestamp()) - int(starttime.timestamp()))/60

    return round(time_diff,2)

#②获取医生诊断的时间
def get_first_diag_time(df_sys_log_id,df_doctor_diag_id):
    df_diag_id = df_doctor_diag_id[df_doctor_diag_id['operation'] != '点击下一个患者时自动保存信息']
    df_diag_id = df_diag_id[df_diag_id['final_diag'].isnull()]

    df_diag_last = df_diag_id.sort_values(by='time',ascending=False)

    if len(df_diag_last) == 0:
        openerate_set = df_doctor_diag_id['operation']
        if '修改了初步诊断' in openerate_set:
            df = df_doctor_diag_id[df_doctor_diag_id['operation'] == '修改了初步诊断']
        else:
            df = df_doctor_diag_id[df_doctor_diag_id['primary_diag'].notnull()]
        endtime = pd.to_datetime(convert_datatime(df.iloc[0]['time_text']))
    else:
        endtime = pd.to_datetime(convert_datatime(df_diag_last.iloc[0]['time_text']))

    df_sys_log_id.loc[:, 'create_time'] = pd.to_datetime(df_sys_log_id['create_time'])
    df_sys_log_id = df_sys_log_id[df_sys_log_id['create_time'] > endtime-pd.Timedelta(hours=1)]
    df_sys_log_id = df_sys_log_id[df_sys_log_id['module'] != '注销登录']

    df_diag_last = df_sys_log_id.sort_values(by='create_time', ascending=True)

    starttime = df_diag_last.iloc[0]['create_time']

    #By the minute.
    time_diff = (int(endtime.timestamp()) - int(starttime.timestamp()))/60

    return round(time_diff,2)

def diag_sequence(df_doctor_diag, doc_pat_id):
    doc_id = doc_pat_id[0]
    pat_id = doc_pat_id[1]

    df_doctor_diag_perdoc = df_doctor_diag[df_doctor_diag['doctor_id'] == doc_id]
    df_doctor_diag_perdoc = df_doctor_diag_perdoc.drop_duplicates(subset=['doctor_id', 'patient_id'])
    df_doctor_diag_perdoc = df_doctor_diag_perdoc.sort_values(by='id', ascending=True)
    df_doctor_diag_perdoc = df_doctor_diag_perdoc.assign(diag_sequence=range(1, len(df_doctor_diag_perdoc) + 1))
    # print(df_doctor_diag_perdoc)

    df_doctor_diag_pat = df_doctor_diag_perdoc[df_doctor_diag_perdoc['patient_id'] == pat_id]
    diag_sequence = df_doctor_diag_pat.iloc[0]['diag_sequence']
    # print(f'diag_sequence {diag_sequence}')
    return diag_sequence


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


def combine_feat(df_sample_id,df_docinfo_id,model_sort, model_visible, model_pre, model_prob_0h, model_prob_3h,diag_seq,first_diag,final_diag,first_diag_time,final_diag_time,percent_action):

    del_list = ['SUBJECT_ID','GROUP', '基础信息（当前）中补充的数据', '下一步检查（当前）中补充的数据', '医生ID']
    for del_sub in del_list:
        del df_sample_id[del_sub]
    df_sample_id['uuid'] = uuid.uuid4()

    df_sample_id['model_sort'] = model_sort
    df_sample_id['model_visible'] = model_visible
    df_sample_id['model_pre'] = model_pre
    df_sample_id['model_prob_0h'] = model_prob_0h
    df_sample_id['model_prob_3h'] = model_prob_3h
    df_sample_id['diag_seq'] = diag_seq
    df_sample_id['first_diag'] = first_diag
    df_sample_id['final_diag'] = final_diag
    df_sample_id['first_diag_time'] = first_diag_time
    df_sample_id['final_diag_time'] = final_diag_time
    df_sample_id['next_act_percent_label'] = percent_action

    if len(df_docinfo_id) == 0:
        return df_sample_id
    docinfo = df_docinfo_id.iloc[0]
    df_sample_id['doctor_logid'] = docinfo['doctor_logid']
    df_sample_id['doctor_unit'] = docinfo['institution_level']
    df_sample_id['doctor_sex'] = docinfo['gender']
    df_sample_id['doctor_age'] = docinfo['age']
    df_sample_id['doctor_year'] = docinfo['years_worked']
    df_sample_id['doctor_depart'] = docinfo['department']
    df_sample_id['doctor_title'] = docinfo['class_of_position']
    df_sample_id['doctor_field'] = docinfo['area_of_expertise']

    return df_sample_id


def log_to_csv():
    #The patient id of 1w+ is test data.
    df_syslog = pd.read_csv(root + 'sys_log.csv',encoding='gbk')
    df_syslog = df_syslog[(df_syslog['patient_id'] <= 6000) | (df_syslog['patient_id'] > 20000)]

    df_patient_check = pd.read_csv(root + 'patient_check.csv',encoding='gbk')
    df_patient_check = df_patient_check[(df_patient_check['patient_id'] <= 6000) | (df_patient_check['patient_id'] > 20000)]

    df_doctor_diag = pd.read_csv(root + 'doctor_diag.csv',encoding='gbk')
    df_doctor_diag = df_doctor_diag[(df_doctor_diag['patient_id'] <= 6000) | (df_doctor_diag['patient_id'] > 20000)]

    df_sample = pd.read_csv(root+'sample_patient_model.csv', encoding='gbk')

    df_docinfo = pd.read_csv(root + 'doctor_info.csv', usecols=['doctor_logid','institution_level','gender','age','years_worked','department','class_of_position','area_of_expertise'],encoding='gbk')

    df_syslog_group = df_syslog.groupby(['accountname', 'patient_id'])
    df_patient_check_group = df_patient_check.groupby(['doctor_id', 'patient_id'])
    df_doctor_diag_group = df_doctor_diag.groupby(['doctor_id', 'patient_id'])

    doctor_patient_set1 = set()
    doctor_patient_set2 = set()
    for index, row_doctor_diag in df_doctor_diag_group:
        doctor_patient_set1.add((index[0], index[1]))
    for index, row_doctor_diag in df_syslog_group:
        doctor_patient_set2.add((index[0], index[1]))

    doctor_patient_set = doctor_patient_set1 & doctor_patient_set2
    print(len(doctor_patient_set))

    # ③确定完成诊断的患者列表patient_set
    patient_set = set()
    for item in doctor_patient_set:
        patient_set.add(item[1])

    final_train_data(df_doctor_diag, df_syslog_group, df_patient_check_group, df_doctor_diag_group,
                     doctor_patient_set, df_sample, df_docinfo,to_file)



root = f'{pro_path}datasets/Original-Recorded-Version/'

to_file = f'{pro_path}datasets/csv_and_pkl/data_0321_7000.csv'
if not os.path.exists(f'{pro_path}datasets/csv_and_pkl'):
    os.makedirs(f'{pro_path}datasets/csv_and_pkl')


if __name__ == '__main__':
    log_to_csv()
    # train_test_val()



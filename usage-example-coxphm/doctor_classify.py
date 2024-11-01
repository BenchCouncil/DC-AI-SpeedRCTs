import pandas as pd


def abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag):
    test_uuid_new = []
    test_patient_id_new = []
    test_truedoc_diag_new = []
    test_virdoc_diag_new = []

    for index, uuid in enumerate(test_uuid):
        if uuid in uuid_list:
            patid = test_patient_id[index]
            truedoc_diag = test_truedoc_diag[index]
            virdoc_diag = test_virdoc_diag[index]

            test_uuid_new.append(uuid)
            test_patient_id_new.append(patid)
            test_truedoc_diag_new.append(truedoc_diag)
            test_virdoc_diag_new.append(virdoc_diag)
    combine_param = (test_uuid_new, test_patient_id_new, test_truedoc_diag_new, test_virdoc_diag_new)
    return combine_param


#患者分类，脓毒症患者和非脓毒症患者
def patient_exper_3h_label0(patient_flag,df_sample,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param

    if patient_flag == 'sepsis':
        df = df_sample[df_sample['ILL_TIME'].notnull()]
        df = df[df['TIME_RANGE'] != '3h']
    else:
        df = df_sample[(df_sample['TIME_RANGE'].isnull()) | (df_sample['TIME_RANGE'] == '3h')]

    sepsis_patientid = list(df['UNIQUE_ID'])
    test_uuid_new = []
    test_patient_id_new = []
    test_truedoc_diag_new = []
    test_virdoc_diag_new = []

    for index,uuid in enumerate(test_uuid):
        patid = test_patient_id[index]
        if patid in sepsis_patientid:
            truedoc_diag = test_truedoc_diag[index]
            virdoc_diag = test_virdoc_diag[index]

            test_uuid_new.append(uuid)
            test_patient_id_new.append(patid)
            test_truedoc_diag_new.append(truedoc_diag)
            test_virdoc_diag_new.append(virdoc_diag)
    return test_uuid_new,test_patient_id_new,test_truedoc_diag_new,test_virdoc_diag_new


def patient_exper_3h_label1(patient_flag,df_sample,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param

    if patient_flag == 'sepsis':
        df = df_sample[df_sample['ILL_TIME'].notnull()]
    else:
        df = df_sample[df_sample['TIME_RANGE'].isnull()]

    sepsis_patientid = list(df['UNIQUE_ID'])
    test_uuid_new = []
    test_patient_id_new = []
    test_truedoc_diag_new = []
    test_virdoc_diag_new = []

    for index,uuid in enumerate(test_uuid):
        patid = test_patient_id[index]
        if patid in sepsis_patientid:
            truedoc_diag = test_truedoc_diag[index]
            virdoc_diag = test_virdoc_diag[index]

            test_uuid_new.append(uuid)
            test_patient_id_new.append(patid)
            test_truedoc_diag_new.append(truedoc_diag)
            test_virdoc_diag_new.append(virdoc_diag)
    return test_uuid_new,test_patient_id_new,test_truedoc_diag_new,test_virdoc_diag_new

def model_exper(model_sort,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param
    if model_sort == '无模型':
        df = df_log[~df_log['model_sort'].notna()]
    else:
        df = df_log[df_log['model_sort'].notna()]
        df = df[df['model_sort'].str.contains(model_sort)]

    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)

def model_visexper(model_sort,model_vis,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param
    df = df_log[df_log['model_sort'].notna()] #只统计有模型的情况下

    if model_sort == None:
        df = df[df['model_visible'] == model_vis]
    else:
        df = df[(df['model_sort'].str.contains(model_sort)) & (df['model_visible'] == model_vis)]

    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)


def doctor_unit(unit,df_log,combine_param):
    if unit == '二甲':
        df = df_log[df_log['doctor_unit'] == 1]
    elif unit == '三甲':
        df = df_log[df_log['doctor_unit'] == 2]
    else:
        df = df_log[df_log['doctor_unit'] == 3]
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)

def doctor_sex(sex,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param
    df = df_log[df_log['doctor_sex'] == sex]
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)

def doctor_age(minage,maxage,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param
    df = df_log[(df_log['doctor_age'] <= maxage) & (df_log['doctor_age'] > minage)]
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)

def doctor_year(min_year,max_year,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param

    df_log = df_log[df_log['doctor_year'].notnull()]
    df = df_log[(df_log['doctor_year'] <= max_year) & (df_log['doctor_year'] > min_year)]
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)

#医生职称
def doctor_title(title,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param

    df_log = df_log[df_log['doctor_title'].notnull()]
    if title == '无':
        df = df_log[df_log['doctor_title'] == '无']
    elif title == '住院医生':
        df = df_log[(df_log['doctor_title'] == '住院医师') | (df_log['doctor_title'] == '初级')]
    elif title == '主治医生':
        df = df_log[
            (df_log['doctor_title'] == '主治') | (df_log['doctor_title'] == '主治医师') | (df_log['doctor_title'] == '主治医生')]
    else :
        #title == '副/主任医生':
        df = df_log[(df_log['doctor_title'] == '副主任医师') | (df_log['doctor_title'] == '副主任医生') | (
                    df_log['doctor_title'] == '副主任')| (df_log['doctor_title'] == '主任医师')]

    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)

def doctor_depart(depart_list,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param
    df = df_log[df_log['doctor_depart'].isin(depart_list)]
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list, test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag)
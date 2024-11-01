import pandas as pd
import math
import sys
from project_path import pro_path
from simulator.utils.utils_dataloader import convert_datatime,df_convert_datatime



dict_check_type = {
    '1':'血常规',
    '2': '动脉血气分析',
    '3': '止凝血',
    '4': '影像检查',
    '5': '病原检查',
    '6': '培养',
    '7': '涂片',
    '8': '历史用药',
    '9': '降钙素原',
}

dict_click_num = {
    '历史基础信息':0,
    '既往病史': 1,
}
dict_history_num = {
    '血常规': 2,
    '动脉血气分析': 3,
    '止凝血': 4,
    '影像检查': 5,
    '病原检查': 6,
    '培养': 7,
    '涂片': 8,
    '历史用药': 9,
}
dict_current_num = {
    '降钙素原': 10,
    '血常规': 11,
    '动脉血气分析': 12,
    '止凝血': 13,
    '影像检查': 14,
    '病原检查': 15,
    '培养': 16,
    '涂片': 17,
}

def get_checktype(s):
    start_index = s.find("checkType:")
    if start_index != -1:
        # 找到第一个 "&" 的位置
        end_index = s.find("&", start_index)
        check_type = s[start_index + len("checkType:"):end_index]
        return dict_check_type.get(check_type)
    else:
        return None

path = f'{pro_path}datasets/Original-Recorded-Version/'
df_syslog = pd.read_csv(path+'sys_log.csv',usecols=['accountname','patient_id','module','exception','create_time'],encoding='gbk')
df_syslog['exam_type'] = df_syslog['exception'].apply(lambda x: get_checktype(str(x)))
df_syslog['module'] = df_syslog['module'].apply(lambda x: "既往病史" if str(x) == "历史病例" else x)
df_syslog = df_syslog[df_syslog['module'].isin(['历史基础信息', '既往病史', '历史检查'])] #syslog里面下一步检查记录的有缺失
df_syslog.rename(columns={'accountname': 'doctor_id', 'create_time': 'time_text'}, inplace=True)
del df_syslog['exception']

df_nextcheck = pd.read_csv(path+'patient_check.csv',usecols=['doctor_id','patient_id','exam_type','time_text'],encoding='gbk')
df_nextcheck['module'] = '下一步检查'
df = pd.concat([df_syslog,df_nextcheck])
df = df_convert_datatime(df, 'time_text')
df['time_text'] = pd.to_datetime(df['time_text'])
df_group = df.groupby(['doctor_id', 'patient_id'])


def get_click_seq(doctorid,patientid):
    if str(doctorid) == 'nan':
        return None
    if (doctorid, patientid) not in df_group.groups.keys():
        patientid = patientid + 20000
        if (doctorid, patientid) not in df_group.groups.keys():
            # print(f'医生id {doctorid}，患者id {patientid}没有查看检查项的轨迹信息')
            return None
    df_row = df_group.get_group((doctorid,patientid))

    df_row = df_row.sort_values('time_text',ascending=True)
    df_row = df_row.drop_duplicates()
    df_row.loc[df_row['module'].isin(['历史基础信息', '既往病史']), 'click_seq'] = df_row['module'].map(dict_click_num)
    df_row.loc[df_row['module'].isin(['历史检查']), 'click_seq'] = df_row['exam_type'].map(dict_history_num)
    df_row.loc[df_row['module'].isin(['下一步检查']), 'click_seq'] = df_row['exam_type'].map(dict_current_num)
    click_seq = list(df_row['click_seq'])
    click_seq_after = list(filter(lambda x: not math.isnan(x), click_seq))#因为日志中有历史检查 但是没记录具体类型的情况，所以映射为None
    if len(click_seq_after) == 0:
        return None
    if len(click_seq_after) > 18: #判断 医生是否多次 重复诊断这个患者了
        df_row['hour'] = df_row['time_text'].dt.strftime('%Y-%m-%d %H')
        df_row = df_row[df_row['hour'] == df_row.iloc[0]['hour']]
        click_seq = list(df_row['click_seq'])
        click_seq_after = list(filter(lambda x: not math.isnan(x), click_seq))
    return click_seq_after




if __name__ == '__main__':
    for (doc,pa),_ in df_group:
        # if doc == 47 and pa == 4349:
            get_click_seq(doc,pa)
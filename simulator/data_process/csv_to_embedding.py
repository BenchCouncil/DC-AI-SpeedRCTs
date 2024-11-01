import sys
from project_path import pro_path
sys.path.append(pro_path)
import pandas as pd
import re
import ast
import numpy as np
import pickle
from simulator.data_process.embedding.demoinfo_embedding import demoinfo_embedding
from simulator.data_process.embedding.tsfresh_embedding import tsfresh_embedding
from simulator.data_process.embedding.biobert_embedding import biobert_embe
from simulator.data_process.embedding.xray_embedding import multi_chest_xray_embeddings, single_chest_xray_embeddings
import os
from simulator.utils.utils_io_pkl import read_patient_emb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import concurrent.futures


def merge_dataframes(df1, df2):
    merged_df = pd.concat([df1, df2]).drop_duplicates()


    return merged_df


def extract_numbers(text):
    # 使用正则表达式匹配数字（包括小数）
    if '10^9/L' in str(text):
        text = str(text).replace('10^9/L', '')
    if '意识改变' in str(text):
        return [1]
    if '意识正常' in str(text):
        return [0]
    if isinstance(text, int):
        return [text]
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if len(numbers) == 0:
        return [0.00]
    float_numbers = [float(num) for num in numbers]
    return float_numbers


def current_base_todf(row):
    df = pd.DataFrame()
    data = eval(row['基础信息（当前）'])
    for item in data:
        checkup_name = list(item.keys())[0]
        if '体液24h' in checkup_name:
            continue
        time = item[checkup_name]['时间']
        if checkup_name == 'QSOFA':
            value_dict = item[checkup_name]
            for key, value in value_dict.items():
                if key != '收缩压':
                    if key == '时间':
                        df.at[0, f'{key}'] = value
                    else:
                        df.at[0, f'{key}'] = extract_numbers(value)[0]
        elif checkup_name == '血压':
            value = item[checkup_name]['值']
            high_value = value.split('/')[0]
            low_value = value.split('/')[1][:-4]
            df.at[0, '收缩压'] = high_value
            df.at[0, '舒张压'] = low_value
        else:
            value = item[checkup_name]['值']
            df.at[0, checkup_name] = extract_numbers(value)[0]
    df['时间'] = pd.to_datetime(df['时间'])
    return df


def his_todf(row, key):
    data = row[key]
    if str(data) == 'nan':
        return pd.DataFrame()

    if '体温' in key:
        data = ast.literal_eval(data)
        processed_data = [{'时间': entry['时间'], '体温': float(entry['值'][:-2])}
                          for entry in data['体温']]
    elif '心率' in key:
        data = ast.literal_eval(data)
        processed_data = [{'时间': entry['时间'], '心率': float(entry['值'][:-3])}
                          for entry in data['心率']]
    elif '呼吸频率' in key:
        if len(data) == 0:
            return pd.DataFrame()
        processed_data = [{'时间': entry['时间'], '呼吸频率': float(entry['值'][:-8])}
                          for entry in data]
    else:
        if len(data) == 0:
            return pd.DataFrame()
        processed_data = [{'时间': entry['时间'], '意识': extract_numbers(entry['值'])[0]}
                          for entry in data]

    df = pd.DataFrame(processed_data)
    df['时间'] = pd.to_datetime(df['时间'])
    return df


def blood_his_todf(row):
    data = row['基础信息_血压（历史）']
    if str(data) == 'nan':
        return pd.DataFrame()
    data = ast.literal_eval(data)
    processed_data = [
        {'时间': entry['时间'], '收缩压': int(entry['值'].split('/')[0]), '舒张压': int(entry['值'].split('/')[1][:-4])}
        for entry in data['血压']]
    df = pd.DataFrame(processed_data)
    df['时间'] = pd.to_datetime(df['时间'])
    return df


def qsofa_his_todf(row):
    data = row['基础信息_QSOFA（历史）']
    if str(data) == 'nan':
        return pd.DataFrame()
    data_list = eval(data)
    df = pd.DataFrame(columns=['时间'])
    for data_dict in data_list:
        if '呼吸频率' in data_dict.keys():
            df1 = his_todf(data_dict, '呼吸频率')
            if not df1.empty:
                df = pd.merge(df, df1, on=['时间'], how='outer')
        if '意识' in data_dict.keys():
            df2 = his_todf(data_dict, '意识')
            if not df2.empty:
                df = pd.merge(df, df2, on=['时间'], how='outer')
    df['时间'] = pd.to_datetime(df['时间'])
    return df


time_features = set()


def next_todf(data, key):
    flat_data = []
    if str(data) == 'nan' or str(data[key]) == 'nan':
        return pd.DataFrame()
    if isinstance(data[key], dict):
        item = data[key]
        time = item['时间']
        values = {}
        values['时间'] = time
        for value in item['值']:
            key, val = list(value.items())[0]
            key = str(key).split('|')[1]
            val = extract_numbers(val)[0]
            if key != '':
                values[key] = val
                # time_features.add(key)
        flat_data.append(values)
    else:
        for item in data[key]:
            time = item['时间']
            values = {}
            values['时间'] = time
            for value in item['值']:
                key, val = list(value.items())[0]
                key = str(key).split('|')[1]
                val = extract_numbers(val)[0]
                if key != '':
                    values[key] = val
                    # time_features.add(key)
            flat_data.append(values)

    df = pd.DataFrame(flat_data)
    return df


def sofa_todf(row):
    current_endtime = str(row['START_ENDTIME']).split('~')[1]
    current_endtime = pd.to_datetime(current_endtime)
    data_dict = {
        False: 1,
        True: 0
    }
    df = pd.DataFrame()
    sofa1 = ast.literal_eval(row['SOFA_呼吸系统'])
    sofa2 = ast.literal_eval(row['SOFA_凝血系统'])
    sofa3 = ast.literal_eval(row['SOFA_肝脏'])
    sofa4 = ast.literal_eval(row['SOFA_心血管系统'])
    sofa5 = ast.literal_eval(row['SOFA_中枢神经系统'])
    sofa6 = ast.literal_eval(row['SOFA_肾脏'])
    df.at[0, '时间'] = current_endtime
    df.at[0, 'Pao2FiO2'] = extract_numbers(sofa1['Pao2/FiO2'])[0]
    df.at[0, '机械通气'] = data_dict.get(sofa1['机械通气'])
    df.at[0, '血小板'] = extract_numbers(sofa2['血小板'])[0]
    df.at[0, '胆红素'] = extract_numbers(sofa3['胆红素'])[0]
    df.at[0, 'MAP'] = extract_numbers(sofa4['MAP'])[0]
    df.at[0, 'gcs'] = float(sofa5['gcs'])
    df.at[0, '肌酐'] = extract_numbers(sofa6['肌 酐'])[0]

    return df


# 改为2分类
def convert_diag(text):
    label_dict = {
        '无脓毒症': 0,
        '低度疑似脓毒症': 1,
        '高度疑似脓毒症': 1,
        '一般脓毒症': 1,
        '严重脓毒症': 1
    }
    matched_value = None

    # 遍历字典的键
    for key in label_dict:
        if key in text:
            matched_value = label_dict[key]
            break
    return matched_value


# 获取诊断结果的label
def get_diag_label(row):
    first_diag = row['first_diag']
    final_diag = row['final_diag']
    first_diag = convert_diag(first_diag)
    final_diag = convert_diag(final_diag)
    return first_diag, final_diag


def row_to_jpg(row):
    final_jpg_dict = {}
    first_jpg_dict = {}
    admittime = pd.to_datetime(row['ADMITTIME'])
    current_next = row['下一步检查（当前）']
    if str(current_next) != 'nan':
        current_next = eval(current_next)
        for data in current_next:
            if '影像报告' in data.keys():
                value_list = data['影像报告']
                for value in value_list:
                    if '影像图片' in value.keys() and '时间' in value.keys():
                        jpg = value.get('影像图片')
                        deltatime = pd.to_datetime(value.get('时间')) - admittime
                        hours, _ = divmod(deltatime.total_seconds(), 3600)
                        final_jpg_dict[jpg] = hours

    his_report = row['下一步检查_影像报告（历史）']
    if str(his_report) != 'nan':
        his_report = ast.literal_eval(his_report)
        for report_dict in his_report['影像报告']:
            if '影像图片' in report_dict.keys() and '时间' in report_dict.keys():
                jpg = report_dict.get('影像图片')
                deltatime = pd.to_datetime(report_dict.get('时间')) - admittime
                hours, _ = divmod(deltatime.total_seconds(), 3600)
                final_jpg_dict[jpg] = hours
                first_jpg_dict[jpg] = hours
    return first_jpg_dict, final_jpg_dict


def row_to_notes(row):
    note_dict = {}
    admittime = pd.to_datetime(row['ADMITTIME'])
    current_endtime = str(row['START_ENDTIME']).split('~')[1]
    detaltime = pd.to_datetime(current_endtime) - admittime
    hours, _ = divmod(detaltime.total_seconds(), 3600)
    old_ill_his = ''
    # if str(row['病史']) != 'nan':
    #     old_ill_his = row['病史'].replace('\n', ',')
    #     note_dict[old_ill_his] = hours

    current_illhis = row['现病史']
    if str(current_illhis) != 'nan':
        index1 = str(current_illhis).find('现病史（中文）:')
        index2 = str(current_illhis).find('现病史（原英文）:')
        current_illhis = current_illhis[index2 + 9:]
        current_illhis = current_illhis.replace('_', '')
        current_illhis = current_illhis.replace('\n', '')
        note_dict[current_illhis] = hours

    chief_comp = row['主诉']
    if str(chief_comp) != 'nan':
        index1 = str(chief_comp).find('中文：')
        index2 = str(chief_comp).find('英文：')
        chief_comp = chief_comp[index2 + 3:]
        chief_comp = chief_comp.replace('\n', '')
        note_dict[chief_comp] = hours
    first_note_dict = note_dict
    final_note_dict = note_dict

    current_next = row['下一步检查（当前）']
    if str(current_next) != 'nan':
        current_next = eval(current_next)
        for data in current_next:
            if '影像报告' in data.keys():
                value_list = data['影像报告']

                for value_dict in value_list:
                    time = value_dict['时间']
                    value = value_dict['值']
                    value = value.replace('\n', '')
                    value = value.replace('_', '')
                    value = value.replace('  ', '')

                    if '中文报告：' in str(value):
                        index_en = str(value).find('英文报告：')
                        value = value[index_en + 5:]
                    deltatime = pd.to_datetime(time) - admittime
                    hours, _ = divmod(deltatime.total_seconds(), 3600)
                    final_note_dict[value] = hours

    his_report = row['下一步检查_影像报告（历史）']
    if str(his_report) != 'nan':
        his_report = ast.literal_eval(his_report)
        for report_dict in his_report['影像报告']:
            time = report_dict['时间']
            value = report_dict['值']
            value = value.replace('\n', '')
            value = value.replace('_', '')
            value = value.replace('  ', '')
            if '中文报告：' in str(value):
                index_en = str(value).find('英文报告：')
                value = value[index_en + 5:]

            deltatime = pd.to_datetime(time) - admittime
            hours, _ = divmod(deltatime.total_seconds(), 3600)
            first_note_dict[value] = hours
            final_note_dict[value] = hours
    return first_note_dict, final_note_dict


def row_to_timeseries(row):
    df_base = current_base_todf(row)
    df_his1 = his_todf(row, '基础信息_体温（历史）')
    df_his2 = his_todf(row, '基础信息_心率（历史）')
    df_his3 = blood_his_todf(row)
    df_his4 = qsofa_his_todf(row)
    df_first_time_series = df_base
    if not df_his1.empty:
        df_first_time_series = merge_dataframes(df_first_time_series, df_his1)
    if not df_his2.empty:
        df_first_time_series = merge_dataframes(df_first_time_series, df_his2)
    if not df_his3.empty:
        df_first_time_series = merge_dataframes(df_first_time_series, df_his3)
    if not df_his4.empty:
        df_first_time_series = merge_dataframes(df_first_time_series, df_his4)
    # 合并同一时刻的所有值

    if str(row['下一步检查_血常规（历史）']) != 'nan':
        df_his5 = next_todf(ast.literal_eval(row['下一步检查_血常规（历史）']), '血常规')
        df_first_time_series = merge_dataframes(df_first_time_series, df_his5)
    if str(row['下一步检查_动脉血气分析（历史）']) != 'nan':
        df_his6 = next_todf(ast.literal_eval(row['下一步检查_动脉血气分析（历史）']), '动脉血气分析')
        df_first_time_series = merge_dataframes(df_first_time_series, df_his6)
    if str(row['下一步检查_止凝血（历史）']) != 'nan':
        df_his7 = next_todf(ast.literal_eval(row['下一步检查_止凝血（历史）']), '止凝血')
        df_first_time_series = merge_dataframes(df_first_time_series, df_his7)

    # df_first_time_series = df_first_time_series.groupby('时间').apply(lambda group: group.ffill().bfill()).reset_index(
    #     drop=True)
    df_first_time_series = df_first_time_series.groupby('时间').apply(
        lambda group: group.ffill().bfill().reset_index(drop=True)).reset_index(drop=True)

    df_first_time_series = df_first_time_series.drop_duplicates(subset=['时间'])

    df_final_time_series = df_first_time_series

    df_next = pd.DataFrame()
    for current_next_dict in eval(row['下一步检查（当前）']):
        df_temp = pd.DataFrame()
        if '血常规' in current_next_dict.keys():
            df_temp = next_todf(current_next_dict, '血常规')
        if '动脉血气分析' in current_next_dict.keys():
            df_temp = next_todf(current_next_dict, '动脉血气分析')
        if '止凝血' in current_next_dict.keys():
            df_temp = next_todf(current_next_dict, '止凝血')
        if not df_temp.empty:
            df_next = merge_dataframes(df_next, df_temp)
    df_final_time_series = merge_dataframes(df_final_time_series, df_next)
    df_sofa = sofa_todf(row)
    df_final_time_series = merge_dataframes(df_final_time_series, df_sofa)

    df_final_time_series = df_final_time_series.groupby('时间').apply(lambda group: group.ffill().bfill()).reset_index(
        drop=True)

    df_final_time_series = df_final_time_series.drop_duplicates(subset=['时间'])
    df_first_time_series['UNIQUE_ID'] = row['UNIQUE_ID']
    df_final_time_series['UNIQUE_ID'] = row['UNIQUE_ID']

    return df_first_time_series, df_final_time_series


def row_to_embedding(row):

    demoinfo_emb = demoinfo_embedding(row)
    df_first_time_series, df_final_time_series = row_to_timeseries(row)
    first_note_dict, final_note_dict = row_to_notes(row)
    first_jpg_dict, final_jpg_dict = row_to_jpg(row)

    first_time_series = tsfresh_embedding(df_first_time_series)
    final_time_series = tsfresh_embedding(df_final_time_series)
    if len(first_jpg_dict.keys()) > 1:
        first_xray_dense_emb, first_xray_predict_emb = multi_chest_xray_embeddings(list(first_jpg_dict.keys()),
                                                                                   list(first_jpg_dict.values()))
    elif len(first_jpg_dict.keys()) == 0:
        first_xray_dense_emb = None
        first_xray_predict_emb = None
    else:
        first_xray_dense_emb, first_xray_predict_emb = single_chest_xray_embeddings(list(first_jpg_dict.keys())[0])

    if len(final_jpg_dict.keys()) > 1:
        final_xray_dense_emb, final_xray_predict_emb = multi_chest_xray_embeddings(list(final_jpg_dict.keys()),
                                                                                   list(final_jpg_dict.values()))
    elif len(final_jpg_dict.keys()) == 0:
        final_xray_dense_emb = None
        final_xray_predict_emb = None
    else:
        final_xray_dense_emb, final_xray_predict_emb = single_chest_xray_embeddings(list(final_jpg_dict.keys())[0])

    first_note_emb = biobert_embe(list(first_note_dict.keys()), list(first_note_dict.values()))
    final_note_emb = biobert_embe(list(final_note_dict.keys()), list(final_note_dict.values()))

    df_demoinfo_embeddings_fusion = pd.DataFrame(demoinfo_emb.reshape(1, -1),
                                                 columns=['demoinfo_' + str(i) for i in range(demoinfo_emb.shape[0])])
    df_first_ts_embeddings_fusion = pd.DataFrame(first_time_series.values.reshape(1, -1),
                                                 columns=['first_ts_' + str(i) for i in
                                                          range(first_time_series.values.shape[0])])
    df_final_ts_embeddings_fusion = pd.DataFrame(final_time_series.values.reshape(1, -1),
                                                 columns=['final_ts_' + str(i) for i in
                                                          range(final_time_series.values.shape[0])])

    if first_xray_dense_emb is not None:
        df_first_xray_dense_embe_fusion = pd.DataFrame(first_xray_dense_emb.reshape(1, -1),
                                                       columns=['xrayd_' + str(i) for i in
                                                                range(first_xray_dense_emb.shape[0])])
        df_first_xray_predict_embed_fusion = pd.DataFrame(first_xray_predict_emb.reshape(1, -1),
                                                          columns=['xrayp_' + str(i) for i in
                                                                   range(first_xray_predict_emb.shape[0])])
    else:
        df_first_xray_dense_embe_fusion = pd.DataFrame(np.zeros((1, 1024)),
                                                       columns=['xrayd_' + str(i) for i in range(1024)])
        df_first_xray_predict_embed_fusion = pd.DataFrame(np.zeros((1, 18)),
                                                          columns=['xrayp_' + str(i) for i in range(18)])

    if final_xray_dense_emb is not None:
        df_final_xray_dense_embe_fusion = pd.DataFrame(final_xray_dense_emb.reshape(1, -1),
                                                       columns=['xrayd_' + str(i) for i in
                                                                range(final_xray_dense_emb.shape[0])])
        df_final_xray_predict_embed_fusion = pd.DataFrame(final_xray_predict_emb.reshape(1, -1),
                                                          columns=['xrayp_' + str(i) for i in
                                                                   range(final_xray_predict_emb.shape[0])])
    else:
        df_final_xray_dense_embe_fusion = pd.DataFrame(np.zeros((1, 1024)),
                                                       columns=['xrayd_' + str(i) for i in range(1024)])
        df_final_xray_predict_embed_fusion = pd.DataFrame(np.zeros((1, 18)),
                                                          columns=['xrayp_' + str(i) for i in range(18)])

    df_first_note_embeddings_fusion = pd.DataFrame(first_note_emb.reshape(1, -1),
                                                   columns=['note_' + str(i) for i in range(first_note_emb.shape[0])])
    df_final_note_embeddings_fusion = pd.DataFrame(final_note_emb.reshape(1, -1),
                                                   columns=['note_' + str(i) for i in range(final_note_emb.shape[0])])

    df_haim_ids_fusion = pd.DataFrame([row['UNIQUE_ID']], columns=['haim_id'])
    df_fusion_first = df_haim_ids_fusion
    df_fusion_final = df_haim_ids_fusion

    df_fusion_first = pd.concat([df_fusion_first, df_demoinfo_embeddings_fusion], axis=1)
    df_fusion_first = pd.concat([df_fusion_first, df_first_ts_embeddings_fusion], axis=1)
    df_fusion_first = pd.concat([df_fusion_first, df_first_xray_dense_embe_fusion], axis=1)
    df_fusion_first = pd.concat([df_fusion_first, df_first_xray_predict_embed_fusion], axis=1)
    df_fusion_first = pd.concat([df_fusion_first, df_first_note_embeddings_fusion], axis=1)

    df_fusion_final = pd.concat([df_fusion_final, df_demoinfo_embeddings_fusion], axis=1)
    df_fusion_final = pd.concat([df_fusion_final, df_final_ts_embeddings_fusion], axis=1)
    df_fusion_final = pd.concat([df_fusion_final, df_final_xray_dense_embe_fusion], axis=1)
    df_fusion_final = pd.concat([df_fusion_final, df_final_xray_predict_embed_fusion], axis=1)
    df_fusion_final = pd.concat([df_fusion_final, df_final_note_embeddings_fusion], axis=1)

    del df_fusion_first['haim_id']
    del df_fusion_final['haim_id']

    return df_fusion_first, df_fusion_final


def write_pkl(fn):
    df = pd.read_csv(fn, encoding='gbk')
    df = df.drop_duplicates(subset=['UNIQUE_ID'])

    patient_embedding = root+'patient_embedding_first.pkl'

    if os.path.exists(patient_embedding):
        unique_id_list, patient_embedding_list = read_patient_emb(patient_embedding)
        df = df[~df['UNIQUE_ID'].isin(unique_id_list)]
    print(f'The number of patients to be embedded is {len(df)}')

    rows = list(df.iterrows())
    row_list = [(index, row) for index, row in rows]

    def process_row(args):
        index, row = args
        print(f'Patient number {index} ,Patient demographics, cxr jpg, image text, timing check data is embedding.')
        unique_id = row['UNIQUE_ID']
        first_diag = convert_diag(row['first_diag'])
        first_diag_time = row['first_diag_time']
        final_diag = convert_diag(row['final_diag'])
        final_diag_time = row['final_diag_time']
        if final_diag is None or first_diag is None:
            return None
        first_cat_embedding,final_cat_embedding = row_to_embedding(row)
        if first_cat_embedding.shape != (1, 2474) or final_cat_embedding.shape != (1,2474):
            print(f'第{index}行 诊断和诊断时间的embedding')
            print(f'first shape {first_cat_embedding.shape}')
            print(f'final shape {final_cat_embedding.shape}')
            print('此患者缺少信息，不添加到训练集中')
        else:
            first_to_save = (unique_id, first_cat_embedding)
            with open(root+f'patient_embedding_first.pkl', 'ab') as file: # 追加写入
                pickle.dump(first_to_save, file)

            final_to_save = (unique_id, final_cat_embedding)
            with open(root+f'patient_embedding_final.pkl', 'ab') as file: # 追加写入
                pickle.dump(final_to_save, file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(process_row, row_list))



if __name__ == '__main__':
    root = f'{pro_path}datasets/csv_and_pkl/'

    data = root + 'data_0321_7000.csv'

    write_pkl(data)

    print('-----Patient information embedding complete!------')


import pandas as pd
import numpy as np


model_visible_dict = {
    'Yes':1,
    'No':2
}

model_pre_result_dict = {
    '正常':1,
    '脓毒症':2,
    '脓毒症预警':3
}

eval_dict = {
    'LSTM_AUC95_sepsis_sen': 0.9723,
    'LSTM_AUC95_sepsis_spe': 0.8101,
    'LSTM_AUC95_sepsis_acc': 0.8895,
    'LSTM_AUC95_sepsis_auc': 0.9466,

    'LSTM_AUC95_normal_0h_sen': 0.9718,
    'LSTM_AUC95_normal_0h_spe': 0.8058,
    'LSTM_AUC95_normal_0h_acc': 0.8772,
    'LSTM_AUC95_normal_0h_auc': 0.9468,
    'LSTM_AUC95_normal_3h_sen': 0.9728,
    'LSTM_AUC95_normal_3h_spe': 0.8144,
    'LSTM_AUC95_normal_3h_acc': 0.9019,
    'LSTM_AUC95_normal_3h_auc': 0.9465,

    'LSTM_AUC85_sepsis_sen': 0.9424,
    'LSTM_AUC85_sepsis_spe': 0.6666,
    'LSTM_AUC85_sepsis_acc': 0.7945,
    'LSTM_AUC85_sepsis_auc': 0.8501,

    'LSTM_AUC85_normal_0h_sen': 0.9669,
    'LSTM_AUC85_normal_0h_spe': 0.5667,
    'LSTM_AUC85_normal_0h_acc': 0.7388,
    'LSTM_AUC85_normal_0h_auc': 0.8497,
    'LSTM_AUC85_normal_3h_sen': 0.9180,
    'LSTM_AUC85_normal_3h_spe': 0.7666,
    'LSTM_AUC85_normal_3h_acc': 0.8502,
    'LSTM_AUC85_normal_3h_auc': 0.8506,

    'LSTM_AUC75_sepsis_sen': 0.8941,
    'LSTM_AUC75_sepsis_spe': 0.4759,
    'LSTM_AUC75_sepsis_acc': 0.689,
    'LSTM_AUC75_sepsis_auc': 0.7506,

    'LSTM_AUC75_normal_0h_sen': 0.8089,
    'LSTM_AUC75_normal_0h_spe': 0.5149,
    'LSTM_AUC75_normal_0h_acc': 0.6414,
    'LSTM_AUC75_normal_0h_auc': 0.7447,
    'LSTM_AUC75_normal_3h_sen': 0.9793,
    'LSTM_AUC75_normal_3h_spe': 0.4369,
    'LSTM_AUC75_normal_3h_acc': 0.7366,
    'LSTM_AUC75_normal_3h_auc': 0.7565,

    'TREWScore_sepsis_sen': 0.8528,
    'TREWScore_sepsis_spe': 0.9568,
    'TREWScore_sepsis_acc': 0.9083,
    'TREWScore_sepsis_auc': 0.935,

    'TREWScore_normal_0h_sen': 0.8256,
    'TREWScore_normal_0h_spe': 0.9269,
    'TREWScore_normal_0h_acc': 0.8833,
    'TREWScore_normal_0h_auc': 0.95,
    'TREWScore_normal_3h_sen': 0.88,
    'TREWScore_normal_3h_spe': 0.9867,
    'TREWScore_normal_3h_acc': 0.9333,
    'TREWScore_normal_3h_auc': 0.92,

    'RandomModel_sepsis_sen': 0.5,
    'RandomModel_sepsis_spe': 0.5,
    'RandomModel_sepsis_acc': 0.5,
    'RandomModel_sepsis_auc': 0.5,

    'RandomModel_normal_0h_sen': 0.5,
    'RandomModel_normal_0h_spe': 0.5,
    'RandomModel_normal_0h_acc': 0.5,
    'RandomModel_normal_0h_auc': 0.5,
    'RandomModel_normal_3h_sen': 0.5,
    'RandomModel_normal_3h_spe': 0.5,
    'RandomModel_normal_3h_acc': 0.5,
    'RandomModel_normal_3h_auc': 0.5,

    'LGBM_AUC100_sepsis_sen': 0.9675,
    'LGBM_AUC100_sepsis_spe': 0.9364,
    'LGBM_AUC100_sepsis_acc': 0.9541,
    'LGBM_AUC100_sepsis_auc': 0.9903,

    'LGBM_AUC100_normal_0h_sen': 0.935,
    'LGBM_AUC100_normal_0h_spe': 0.8739,
    'LGBM_AUC100_normal_0h_acc': 0.9086,
    'LGBM_AUC100_normal_0h_auc': 0.9807,
    'LGBM_AUC100_normal_3h_sen': 1.0,
    'LGBM_AUC100_normal_3h_spe': 0.999,
    'LGBM_AUC100_normal_3h_acc': 0.9996,
    'LGBM_AUC100_normal_3h_auc': 1.0,

    'LGBM_AUC90_sepsis_sen': 0.8356,
    'LGBM_AUC90_sepsis_spe': 0.7935,
    'LGBM_AUC90_sepsis_acc': 0.8195,
    'LGBM_AUC90_sepsis_auc': 0.9004,

    'LGBM_AUC90_normal_0h_sen': 0.8351,
    'LGBM_AUC90_normal_0h_spe': 0.7543,
    'LGBM_AUC90_normal_0h_acc': 0.8044,
    'LGBM_AUC90_normal_0h_auc': 0.8892,
    'LGBM_AUC90_normal_3h_sen': 0.8361,
    'LGBM_AUC90_normal_3h_spe': 0.8328,
    'LGBM_AUC90_normal_3h_acc': 0.8346,
    'LGBM_AUC90_normal_3h_auc': 0.9116,

    'LGBM_AUC80_sepsis_sen': 0.7692,
    'LGBM_AUC80_sepsis_spe': 0.7284,
    'LGBM_AUC80_sepsis_acc': 0.754,
    'LGBM_AUC80_sepsis_auc': 0.8347,

    'LGBM_AUC80_normal_0h_sen': 0.7809,#使用的测试集上的评估结果，sepsis为0h和3h的均值
    'LGBM_AUC80_normal_0h_spe': 0.6977,
    'LGBM_AUC80_normal_0h_acc': 0.7498,
    'LGBM_AUC80_normal_0h_auc': 0.8329,
    'LGBM_AUC80_normal_3h_sen': 0.7575,
    'LGBM_AUC80_normal_3h_spe': 0.7591,
    'LGBM_AUC80_normal_3h_acc': 0.7582,
    'LGBM_AUC80_normal_3h_auc': 0.8365,
}

auc_level = {
    '90-100':5,
    '80-90':4,
    '70-80':3,
    '60-70':2,
    '0-60':1
}


model_type_dict = {
    #DL 深度学习
    'LSTM_AUC95': 1,
    'LSTM_AUC85': 1,
    'LSTM_AUC75': 1,
    #ML 机器学习
    'TREWScore': 2,
    'LGBM_AUC80':2,
    'LGBM_AUC90': 2,
    'LGBM_AUC100': 2,
    #随机模型
    'RandomModel': 3
}


# 专用模型    返回10维
#模型种类（DL:lstm、ML:TREWScore、RandomModel）、sen、spe、acc、auc、是否可见、精度类别（根据auc划分四个等级）、预测结果、pro0h、pro3h

##普适模型      返回9维  只有pro0h 或者 pro3h不一样
#模型种类（DL:lstm、ML:TREWScore、RandomModel）、sen、spe、acc、auc、是否可见、精度类别（根据auc划分四个等级）、预测结果、pro0h或者pro3h
def model_embedding(row,sepsis_or_normal):

    sort_value = row['model_sort']
    if sort_value in model_type_dict.keys():
        model_type = model_type_dict.get(sort_value)
        sen = eval_dict.get(f'{sort_value}_{sepsis_or_normal}_sen')
        spe = eval_dict.get(f'{sort_value}_{sepsis_or_normal}_spe')
        acc = eval_dict.get(f'{sort_value}_{sepsis_or_normal}_acc')
        auc = eval_dict.get(f'{sort_value}_{sepsis_or_normal}_auc')
        model_visible = model_visible_dict.get(row['model_visible'])
        model_pre_result = model_pre_result_dict.get(row['model_pre'])
        mode_pre_0h = row['model_prob_0h']
        mode_pre_3h = row['model_prob_3h']
        if auc is None:
            auc_cat = 0
        elif auc >= 0.9:
            auc_cat = auc_level['90-100']
        elif 0.8 <= auc :
            auc_cat = auc_level['80-90']
        elif 0.7 <= auc :
            auc_cat = auc_level['70-80']
        elif 0.6 <= auc:
            auc_cat = auc_level['60-70']
        else:
            auc_cat = auc_level['0-60']
        if sepsis_or_normal == 'sepsis': #专用模型
            embedding = [model_type,sen,spe,acc,auc,model_visible,auc_cat,model_pre_result,mode_pre_0h,mode_pre_3h]
        else:
            if sepsis_or_normal == 'normal_0h':
                embedding = [model_type,sen,spe,acc,auc,model_visible,auc_cat,model_pre_result,mode_pre_0h]
            else: #‘normal_3h’
                embedding = [model_type,sen,spe,acc,auc,model_visible,auc_cat,model_pre_result,mode_pre_3h]
    else:

        # print('模型信息异常')
        if sepsis_or_normal == 'sepsis':
            embedding = [0,0,0,0,0,0,0,0,0,0] #专用模型10维度
        else:
            embedding = [0,0,0,0,0,0,0,0,0] #普适模型9维度

    return np.array(embedding)


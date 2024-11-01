import sys
from project_path import pro_path
sys.path.append(pro_path)
from simulator.utils.utils_io_model import *
from model_name_constant import *
from simulator.utils.utils_dataloader import *
import pandas as pd
from doctor_classify_diagtime import *




def get_diagtime_model(fn_pkl,flag,model_sort,diag_modelname):
    data = Data(fn=fn_pkl, flag=flag)

    test_em, test_label = coxphm_test_diagtime(data,flag)
    test_uuid = data.uuid
    test_patid = data.patient_id

    _, lowest_mae, model = lowest_rmse_model(model_path, flag, diag_modelname)
    print('-----Check the effect evaluation on the coxphm set-----')
    test_y_range = diagtime_label_to_range(test_label, model.predict(test_em), 0.2)
    rmse_eval(test_y_range, model.predict(test_em))

    test_virdoc_diagtime = model.predict(test_em)

    low_diagtime,high_diagtime = diagtime_label_to_range_statis(test_label,0.2)

    return  model,test_uuid,test_patid,test_label,test_y_range,test_virdoc_diagtime,low_diagtime,high_diagtime


def first_virdoc_diagtime():
    fn_pkl = model_input_path + f'/first_data_7000_dim_{feature_dimnum_first}_coxphm.pkl'
    _,test_uuid,test_patid,test_truedoc_diag,test_truedoc_diag_range,test_virdoc_diag,low_diagtime,high_diagtime = get_diagtime_model(fn_pkl,'first','diagtime',first_diagtime_modelname)
    combine_param = (test_uuid,test_patid,test_truedoc_diag,test_truedoc_diag_range,test_virdoc_diag)
    return combine_param,low_diagtime,high_diagtime

def final_virdoc_diagtime():
    fn_pkl = model_input_path + f'/final_data_7000_dim_{feature_dimnum_final}_coxphm.pkl'
    _,test_uuid,test_patid,test_truedoc_diag,test_truedoc_diag_range,test_virdoc_diag,low_diagtime,high_diagtime = get_diagtime_model(fn_pkl,'final','diagtime',final_diagtime_modelname)
    combine_param = (test_uuid,test_patid,test_truedoc_diag,test_truedoc_diag_range,test_virdoc_diag)
    return combine_param,low_diagtime,high_diagtime

def get_true_virdoc_acc(log,df_sample,combine_param,truedoc_low_diagtime,truedoc_high_diagtime):
    truedoc_diagtime_list = []
    truedoc_diagtime_range_list = []
    truedoc_low_diagtime_list = []
    truedoc_high_diagtime_list = []

    virdoc_diagtime_list = []
    test_uuid, test_patid,test_truedoc_diagtime,test_truedoc_diagtime_range, test_virdoc_diagtime =combine_param
    for uuid in test_uuid:
        truedoc = test_truedoc_diagtime[test_uuid.index(uuid)]
        truedoc_range = test_truedoc_diagtime_range[test_uuid.index(uuid)]

        virdoc = test_virdoc_diagtime[test_uuid.index(uuid)]

        truedoc_diagtime_list.append(truedoc)
        truedoc_diagtime_range_list.append(truedoc_range)
        virdoc_diagtime_list.append(virdoc)

        truedoc_low = truedoc_low_diagtime[test_uuid.index(uuid)]
        truedoc_high = truedoc_high_diagtime[test_uuid.index(uuid)]

        truedoc_low_diagtime_list.append(truedoc_low)
        truedoc_high_diagtime_list.append(truedoc_high)

    truedoc_avgenext = None
    if len(truedoc_diagtime_list) != 0:
        truedoc_avgenext = round(sum(truedoc_diagtime_list)/len(truedoc_diagtime_list),2)
    truedoc_range_avgenext = None
    if len(truedoc_diagtime_list) != 0:
        truedoc_range_avgenext = round(sum(truedoc_diagtime_range_list)/len(truedoc_diagtime_range_list),2)

    virdoc_avgenext = None
    if len(virdoc_diagtime_list) != 0:
        virdoc_avgenext = round(sum(virdoc_diagtime_list)/len(virdoc_diagtime_list),2)

    truedoc_low_avgenext = None
    if len(truedoc_low_diagtime_list) != 0:
        truedoc_low_avgenext = round(sum(truedoc_low_diagtime_list) / len(truedoc_low_diagtime_list), 2)

    truedoc_high_avgenext = None
    if len(truedoc_high_diagtime_list) != 0:
        truedoc_high_avgenext = round(sum(truedoc_high_diagtime_list) / len(truedoc_high_diagtime_list), 2)
    # print(f'{log},数据量为{len(truedoc_diagtime_list)}, 真实医生诊断准确率 {round(truedoc_diagtime,2)}%,虚拟医生诊断准确率 {round(virdoc_diagtime,2)}%')
    
    if value == '1':
      print(len(truedoc_diagtime_list))
    elif value =='2':
      print(f'{truedoc_avgenext}')
    elif value == '3':
      print(f'{virdoc_avgenext}')
    elif value == '4':
      print(f'{truedoc_range_avgenext}')
    elif value == '5':
      print(f'[{truedoc_low_avgenext},{truedoc_high_avgenext}]')

# value = sys.argv[1]
value = '1' # 1 2 5 3


if __name__ == '__main__':

    df_sample = pd.read_csv(final_sample_data,encoding='gbk',usecols=['UNIQUE_ID','ILL_TIME','TIME_RANGE'])
    df_log = pd.read_csv(log_data,encoding='gbk')

    if first_or_final == 'first':
        combine_param_test,low_diagtime,high_diagtime = first_virdoc_diagtime()
    else:
        combine_param_test,low_diagtime,high_diagtime = final_virdoc_diagtime()


    get_true_virdoc_acc('所有测试集',df_sample,combine_param_test,low_diagtime,high_diagtime)
    #------------------模型角度------------------
    get_true_virdoc_acc('测试集中模型 TREWScore 可见',df_sample, model_visexper('TREWScore','Yes', df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中模型 TREWScore 不可见',df_sample, model_visexper('TREWScore','No', df_log, combine_param_test),low_diagtime,high_diagtime)

    #------------------医生角度------------------
    get_true_virdoc_acc('测试集中医生性别 男',df_sample, doctor_sex('男', df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生性别 女',df_sample, doctor_sex('女', df_log, combine_param_test),low_diagtime,high_diagtime)

    get_true_virdoc_acc('测试集中医生年龄 小于30（包含30）',df_sample, doctor_age(18,30, df_log,combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生年龄 30-40（包含40）',df_sample, doctor_age(30,40, df_log,combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生年龄 40-50（包含50）',df_sample, doctor_age(40,50, df_log,combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生年龄 50-60（包含60）',df_sample, doctor_age(50,60, df_log,combine_param_test),low_diagtime,high_diagtime)

    get_true_virdoc_acc('测试集中医生医院等级 三甲',df_sample, doctor_unit('三甲', df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生医院等级 二甲',df_sample, doctor_unit('二甲', df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生医院等级 医学院',df_sample, doctor_unit('医学院', df_log, combine_param_test),low_diagtime,high_diagtime)


    get_true_virdoc_acc('测试集中医生从业年限 0-5（包含5）',df_sample, doctor_year(-1,5, df_log,combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生从业年限 5-10（包含10）',df_sample, doctor_year(5,10, df_log,combine_param_test ),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生从业年限 10-15（包15）',df_sample, doctor_year(10,15,df_log,combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生从业年限 15-20（包含20）',df_sample,doctor_year(15,20,df_log,combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生从业年限 大于20',df_sample, doctor_year(20,50,df_log,combine_param_test ),low_diagtime,high_diagtime)

    get_true_virdoc_acc('测试集中医生职称 无',df_sample, doctor_title('无', df_log,combine_param_test ),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生职称 住院医生',df_sample,doctor_title('住院医生', df_log, combine_param_test ),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生职称 主治医生',df_sample,doctor_title('主治医生', df_log, combine_param_test ),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生职称 副/主任医生',df_sample,doctor_title('副/主任医生', df_log,combine_param_test ),low_diagtime,high_diagtime)

    get_true_virdoc_acc('测试集中医生部门 急诊', df_sample, doctor_depart(['急诊','急诊科','急诊内科','急诊医学科'], df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 ICU', df_sample, doctor_depart(['ICU','icu','重症医学二病区','重症医学科二病区','重症医学科一病区','重症医学科','重症科','重症医学','呼吸与危重症医学科'], df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 内科', df_sample, doctor_depart(['内科','肾内科','心内科','呼吸内科','神经内科','心血管内科','综合内科','内分泌科','呼吸'], df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 外科', df_sample, doctor_depart(['神经外科','普外科','心胸血管外科','胸心血管外科（监护室）'], df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 骨科',df_sample, doctor_depart(['骨科'], df_log,combine_param_test ),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 儿科',df_sample,doctor_depart(['儿科'], df_log, combine_param_test ),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 眼科',df_sample,doctor_depart(['眼科'], df_log, combine_param_test ),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 妇科',df_sample,doctor_depart(['妇科','妇产科'], df_log,combine_param_test ),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 中医', df_sample, doctor_depart(['中医科','中医肛肠科','中医康复科'], df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 感染性疾病科', df_sample, doctor_depart(['感染性疾病科'], df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 风湿免疫', df_sample, doctor_depart(['风湿免疫'], df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 脾胃科', df_sample, doctor_depart(['脾胃科'], df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 神经综合科', df_sample, doctor_depart(['神经综合科'], df_log, combine_param_test),low_diagtime,high_diagtime)
    get_true_virdoc_acc('测试集中医生部门 麻醉科', df_sample, doctor_depart(['麻醉科'], df_log, combine_param_test),low_diagtime,high_diagtime)

    #------------------患者角度------------------
    if 'normal_' not in model_path:
        if sepsis_3h_label == 'sepsis_1' :
            get_true_virdoc_acc('测试集中脓毒症患者',df_sample,patient_exper_3h_label1('sepsis',df_sample,combine_param_test),low_diagtime,high_diagtime)
            get_true_virdoc_acc('测试集中正常患者',df_sample,patient_exper_3h_label1('nosepsis',df_sample,combine_param_test),low_diagtime,high_diagtime)
        else:
            get_true_virdoc_acc('测试集中脓毒症患者', df_sample,patient_exper_3h_label0('sepsis', df_sample, combine_param_test),low_diagtime,high_diagtime)
            get_true_virdoc_acc('测试集中正常患者', df_sample,patient_exper_3h_label0('nosepsis', df_sample, combine_param_test),low_diagtime,high_diagtime)

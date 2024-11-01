import sys
from project_path import pro_path
sys.path.append(pro_path)
from simulator.utils.utils_io_model import *
from randomdoc_constant import *
from simulator.utils.utils_dataloader import *
import pandas as pd
from generator.doctor_classify import *




def get_nextact_model(fn_pkl,flag,model_sort,diag_modelname):
    data = Data(fn=fn_pkl, flag=flag)
    _, _, test_em, test_label = data_split(data, flag, model_sort)
    test_uuid,test_patid, test_label_check = sampling_nextact_of_uuid(data)
    if list(test_label) != list(test_label_check):
        print('The uuid of the test set corresponds to the wrong')
    _, best_auc1, model = best_auc_model(model_path, flag, diag_modelname)
    print('-----Check Evaluation of effects on the test set-----')
    test_label = np.array([int(num * 8) for num in test_label])
    multu_class_eval(test_label, model.predict_proba(test_em))

    test_virdoc_next = model.predict(data.embedding)
    return  model,data.uuid,data.patient_id,data.percent_action,test_virdoc_next


def virdoc_nextact():
    fn_pkl = model_input_path + f'/first_data_7000_dim_{feature_dimnum_first}.pkl'

    _,test_uuid,test_patid,test_truedoc_diag,test_virdoc_diag = get_nextact_model(fn_pkl,'first','nextact',first_nextact_modelname)

    combine_param = (test_uuid,test_patid,test_truedoc_diag,test_virdoc_diag)
    return combine_param


def get_true_virdoc_acc(log,df_sample,combine_param):
    truedoc_nextact_list = []

    test_uuid, test_patid,truedoc_nextact, _ =combine_param
    for uuid in test_uuid:
        truedoc = truedoc_nextact[test_uuid.index(uuid)]

        truedoc_nextact_list.append(truedoc)
    truedoc_avgenext = None
    if len(truedoc_nextact_list) != 0:
        truedoc_avgenext = round(sum(truedoc_nextact_list)/len(truedoc_nextact_list),2)

    if value == '1':
      print(len(truedoc_nextact_list))
    elif value =='2':
      print(f'{truedoc_avgenext}')


# value = sys.argv[1]
value = '2'


if __name__ == '__main__':

    df_sample = pd.read_csv(final_sample_data,encoding='gbk',usecols=['UNIQUE_ID','ILL_TIME','TIME_RANGE'])
    df_log = pd.read_csv(log_data,encoding='gbk')
    
    combine_param_test =  virdoc_nextact()

    get_true_virdoc_acc('所有测试集',df_sample,combine_param_test)
    #------------------患者角度------------------
    if 'normal_' not in model_path:#普适模型中没有患者信息
        if sepsis_3h_label == 'sepsis_1':
            get_true_virdoc_acc('测试集中脓毒症患者', df_sample,patient_exper_3h_label1('sepsis', df_sample, combine_param_test))
            get_true_virdoc_acc('测试集中正常患者', df_sample,patient_exper_3h_label1('nosepsis', df_sample, combine_param_test))
        else:
            get_true_virdoc_acc('测试集中脓毒症患者', df_sample,patient_exper_3h_label0('sepsis', df_sample, combine_param_test))
            get_true_virdoc_acc('测试集中正常患者', df_sample,patient_exper_3h_label0('nosepsis', df_sample, combine_param_test))


    #------------------模型角度------------------
    get_true_virdoc_acc('测试集中模型 无模型',df_sample, model_exper('无模型', df_log, combine_param_test))
    get_true_virdoc_acc('测试集中模型 RandomModel',df_sample, model_exper('RandomModel', df_log, combine_param_test))
    get_true_virdoc_acc('测试集中模型 LSTM_AUC75',df_sample, model_exper('LSTM_AUC75', df_log, combine_param_test))
    get_true_virdoc_acc('测试集中模型 LSTM_AUC85',df_sample, model_exper('LSTM_AUC85', df_log, combine_param_test))
    get_true_virdoc_acc('测试集中模型 LSTM_AUC95',df_sample, model_exper('LSTM_AUC95', df_log, combine_param_test))
    get_true_virdoc_acc('测试集中模型 TREWScore',df_sample, model_exper('TREWScore', df_log, combine_param_test))

    #------------------医生角度------------------
    get_true_virdoc_acc('测试集中医生医院等级 三甲',df_sample, doctor_unit('三甲', df_log, combine_param_test))
    get_true_virdoc_acc('测试集中医生医院等级 二甲',df_sample, doctor_unit('二甲', df_log, combine_param_test))
    get_true_virdoc_acc('测试集中医生医院等级 医学院',df_sample, doctor_unit('医学院', df_log, combine_param_test))

    get_true_virdoc_acc('测试集中医生性别 男',df_sample, doctor_sex('男', df_log, combine_param_test))
    get_true_virdoc_acc('测试集中医生性别 女',df_sample, doctor_sex('女', df_log, combine_param_test))

    get_true_virdoc_acc('测试集中医生年龄 小于30（包含30）',df_sample, doctor_age(18,30, df_log,combine_param_test))
    get_true_virdoc_acc('测试集中医生年龄 30-40（包含40）',df_sample, doctor_age(30,40, df_log,combine_param_test))
    get_true_virdoc_acc('测试集中医生年龄 40-50（包含50）',df_sample, doctor_age(40,50, df_log,combine_param_test))
    get_true_virdoc_acc('测试集中医生年龄 50-60（包含60）',df_sample, doctor_age(50,60, df_log,combine_param_test))

    get_true_virdoc_acc('测试集中医生从业年限 0-5（包含5）',df_sample, doctor_year(-1,5, df_log,combine_param_test))
    get_true_virdoc_acc('测试集中医生从业年限 5-10（包含10）',df_sample, doctor_year(5,10, df_log,combine_param_test ))
    get_true_virdoc_acc('测试集中医生从业年限 10-15（包15）',df_sample, doctor_year(10,15,df_log,combine_param_test))
    get_true_virdoc_acc('测试集中医生从业年限 15-20（包含20）',df_sample,doctor_year(15,20,df_log,combine_param_test))
    get_true_virdoc_acc('测试集中医生从业年限 大于20',df_sample, doctor_year(20,50,df_log,combine_param_test ))

    get_true_virdoc_acc('测试集中医生职称 无',df_sample, doctor_title('无', df_log,combine_param_test ))
    get_true_virdoc_acc('测试集中医生职称 住院医生',df_sample,doctor_title('住院医生', df_log, combine_param_test ))
    get_true_virdoc_acc('测试集中医生职称 主治医生',df_sample,doctor_title('主治医生', df_log, combine_param_test ))
    get_true_virdoc_acc('测试集中医生职称 副/主任医生',df_sample,doctor_title('副/主任医生', df_log,combine_param_test ))

    get_true_virdoc_acc('测试集中医生部门 急诊', df_sample, doctor_depart(['急诊','急诊科','急诊内科','急诊医学科'], df_log, combine_param_test))
    get_true_virdoc_acc('测试集中医生部门 ICU', df_sample, doctor_depart(['ICU','icu','重症医学二病区','重症医学科二病区','重症医学科一病区','重症医学科','重症科','重症医学','呼吸与危重症医学科'], df_log, combine_param_test))
    get_true_virdoc_acc('测试集中医生部门 内科', df_sample, doctor_depart(['内科','肾内科','心内科','呼吸内科','神经内科','心血管内科','综合内科','内分泌科','呼吸'], df_log, combine_param_test))
    get_true_virdoc_acc('测试集中医生部门 外科', df_sample, doctor_depart(['神经外科','普外科','心胸血管外科','胸心血管外科（监护室）'], df_log, combine_param_test))
    get_true_virdoc_acc('测试集中医生部门 骨科',df_sample, doctor_depart(['骨科'], df_log,combine_param_test ))
    get_true_virdoc_acc('测试集中医生部门 儿科',df_sample,doctor_depart(['儿科'], df_log, combine_param_test ))
    get_true_virdoc_acc('测试集中医生部门 眼科',df_sample,doctor_depart(['眼科'], df_log, combine_param_test ))
    get_true_virdoc_acc('测试集中医生部门 妇科',df_sample,doctor_depart(['妇科','妇产科'], df_log,combine_param_test ))
    get_true_virdoc_acc('测试集中医生部门 中医', df_sample, doctor_depart(['中医科','中医肛肠科','中医康复科'], df_log, combine_param_test))
    get_true_virdoc_acc('测试集中医生部门 感染性疾病科', df_sample, doctor_depart(['感染性疾病科'], df_log, combine_param_test))
    get_true_virdoc_acc('测试集中医生部门 风湿免疫', df_sample, doctor_depart(['风湿免疫'], df_log, combine_param_test))
    get_true_virdoc_acc('测试集中医生部门 脾胃科', df_sample, doctor_depart(['脾胃科'], df_log, combine_param_test))
    get_true_virdoc_acc('测试集中医生部门 神经综合科', df_sample, doctor_depart(['神经综合科'], df_log, combine_param_test))
    get_true_virdoc_acc('测试集中医生部门 麻醉科', df_sample, doctor_depart(['麻醉科'], df_log, combine_param_test))

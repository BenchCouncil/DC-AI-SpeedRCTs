import sys
from project_path import pro_path
sys.path.append(pro_path)
from transformer import Transformer
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import warnings
warnings.filterwarnings("ignore")
from rouge_metric import PyRouge
from simulator.specialized.click_sequence.trs_constant import *
from simulator.specialized.click_sequence.data_load_test import *
from simulator.specialized.click_sequence.datasets import *

def get_dec_input(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


def get_output(predict,dec_output,flag):
    non_zero_dec_output = dec_output[(dec_output != 0)]
    if 2 in non_zero_dec_output:
        index_2 = (non_zero_dec_output == 2).nonzero(as_tuple=False)[0]
        non_zero_dec_output = non_zero_dec_output[:index_2]
    non_zero_predict = predict[(predict != 0)]
    if 2 in non_zero_predict:
        index_2 = (non_zero_predict == 2).nonzero(as_tuple=False)[0]
        non_zero_predict = non_zero_predict[:index_2]
    label_str_list = [idx2word[n.item()] for n in non_zero_dec_output]
    predict_str_list = [idx2word[n.item()] for n in non_zero_predict]
    if flag == 'test':
        print(f'human clinician click sequence：{label_str_list}')
        print(f'predict click sequence：{predict_str_list}')
    return label_str_list,predict_str_list


def eval_metrics(label_str_list, predict_str_list):
    result_dict = {}

    set_acc_list = []
    true_doc_seqlist = []
    vir_doc_seqlist = []

    for l,p in zip(label_str_list, predict_str_list):
        inter_count = len(set(l).intersection(set(p)))
        set_count = len(set(l))
        set_acc = inter_count / set_count
        set_acc_list.append(set_acc)

        true_doc_seqlist.append(len(l))
        vir_doc_seqlist.append(len(p))


    # range
    label_str_range = [[' '.join(sublist)] for sublist in label_str_list]
    predict_str_range = [' '.join(sublist) for sublist in predict_str_list]

    rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_w=True,
                    rouge_w_weight=rouge_w_weight, rouge_s=True, rouge_su=True, skip_gap=skip_gap)
    rouge_scores = rouge.evaluate(predict_str_range, label_str_range)

    bleu1_balance_list = []
    bleu2_balance_list = []
    bleu3_balance_list = []
    bleu4_balance_list = []
    bleu3_other_list = []
    bleu4_other_list = []

    chencherry = SmoothingFunction()
    for l,p in zip(label_str_list,predict_str_list):
        bleu1_balance = sentence_bleu([l], p, weights=[1],smoothing_function=chencherry.method2)
        bleu2_balance = sentence_bleu([l], p, weights=[0.5, 0.5],smoothing_function=chencherry.method2)
        bleu3_balance = sentence_bleu([l], p, weights=[0.34, 0.3, 0.3],smoothing_function=chencherry.method2)
        bleu4_balance = sentence_bleu([l], p, weights=[0.25, 0.25, 0.25, 0.25],smoothing_function=chencherry.method2)
        bleu3_other = sentence_bleu([l], p, weights=[0.5, 0.25, 0.25],smoothing_function=chencherry.method2)
        bleu4_other = sentence_bleu([l], p, weights=[0.4, 0.2, 0.2, 0.2],smoothing_function=chencherry.method2)
        bleu1_balance_list.append(bleu1_balance)
        bleu2_balance_list.append(bleu2_balance)
        bleu3_balance_list.append(bleu3_balance)
        bleu4_balance_list.append(bleu4_balance)
        bleu3_other_list.append(bleu3_other)
        bleu4_other_list.append(bleu4_other)

    result_dict['set_acc'] = round(sum(set_acc_list)/len(set_acc_list),2)
    result_dict['bleu1_balance'] = round(sum(bleu1_balance_list)/len(bleu1_balance_list),2)
    result_dict['bleu2_balance'] = round(sum(bleu2_balance_list)/len(bleu2_balance_list),2)
    result_dict['bleu3_balance'] = round(sum(bleu3_balance_list)/len(bleu3_balance_list),2)
    result_dict['bleu4_balance'] = round(sum(bleu4_balance_list)/len(bleu4_balance_list),2)
    result_dict['bleu3_other'] = round(sum(bleu3_other_list)/len(bleu3_other_list),2)
    result_dict['bleu4_other'] = round(sum(bleu4_other_list)/len(bleu4_other_list),2)
    result_dict['真实医生点击序列平均长度'] = round(sum(true_doc_seqlist)/len(true_doc_seqlist),2)
    result_dict['预测点击序列平均长度'] = round(sum(vir_doc_seqlist)/len(vir_doc_seqlist),2)

    return rouge_scores,result_dict

rouge_w_weight = 1.2
skip_gap = 1

def eval_of_train(model,enc_inputs,dec_outputs,epoch):
    enc_inputs, dec_outputs = enc_inputs.to(device), dec_outputs.to(device)
    model.to(device)
    model.eval()
    label_str_list_l = []
    predict_str_list_l = []
    label_str_list_len = []
    predict_str_list_len = []

    for enc_input, dec_output in zip(enc_inputs, dec_outputs):
        predict_dec_input = get_dec_input(model, enc_input.view(1, -1), start_symbol=tgt_vocab["S"])
        predict, _, _, _ = model(enc_input.view(1, -1), predict_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]

        label_str_list, predict_str_list = get_output(predict, dec_output, '')
        label_str_list_l.append(label_str_list)
        predict_str_list_l.append(predict_str_list)
        label_str_list_len.append(len(label_str_list))
        predict_str_list_len.append(len(predict_str_list))

    rouge_scores,result_dict = eval_metrics(label_str_list_l,predict_str_list_l)
    rouge_2_f = rouge_scores.get('rouge-2').get('f')
    rouge_l_f = rouge_scores.get('rouge-l').get('f')
    rouge_w_f = rouge_scores.get(f'rouge-w-{rouge_w_weight}').get('f')
    rouge_s_f = rouge_scores.get(f'rouge-s{skip_gap}').get('f')
    set_acc = result_dict.get('set_acc')
    bleu4 = result_dict.get('bleu4_balance')
    truedoc_seqlen = result_dict.get('真实医生点击序列平均长度')
    virdoc_seqlen = result_dict.get('预测点击序列平均长度')
    print(f'----training-epoch:{epoch+1}------------')
    print(f'test evaluation:{result_dict}')
    return rouge_2_f,rouge_l_f,rouge_w_f,rouge_s_f,set_acc,bleu4,truedoc_seqlen,virdoc_seqlen




if __name__ == '__main__':
    model = Transformer().to(device)
    model.load_state_dict(torch.load(model_path + test_model_name))
    model.eval()

    X_train, Y_train, Y_train_uuid, X_test, Y_test, Y_test_uuid, X_coxphm_test, Y_coxphm_test, Y_coxphm_test_uuid = train_test_load(test_resample)
    # for X_temp, Y_temp, Y_uuid, flag in [(X_test, Y_test, Y_test_uuid, 'test')]: #eval in coxphm data
    for X_temp, Y_temp, Y_uuid, flag in [(X_test, Y_test, Y_test_uuid, 'test')]: #eval in test data
        print(f'----------{flag}-----------')

        enc_inputs, dec_inputs, dec_outputs, uuids = make_data(X_temp, Y_temp, Y_uuid)
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)

        label_str_list_l = []
        predict_str_list_l = []
        len_label_list = []
        len_predict_list = []
        index = 1
        for enc_input,dec_output,uuid in zip(enc_inputs,dec_outputs,uuids):
            print(f'-----uuid {uuid}-----')
            index += 1
            predict_dec_input = get_dec_input(model, enc_input.view(1, -1), start_symbol=tgt_vocab["S"])
            predict, _, _, _ = model(enc_input.view(1, -1), predict_dec_input)
            predict = predict.data.max(1, keepdim=True)[1]

            label_str_list,predict_str_list = get_output(predict,dec_output,flag)
            label_str_list_l.append(label_str_list)
            predict_str_list_l.append(predict_str_list)

        rouge_scores,result_dict = eval_metrics(label_str_list_l,predict_str_list_l)

        print(rouge_scores)
        print(result_dict)

import sys
from project_path import pro_path
sys.path.append(pro_path)
import torch
torch.cuda.init()
import torch.nn as nn
from transformer import Transformer
from simulator.specialized.click_sequence.predict_and_eval import eval_of_train
import datetime
import os
from simulator.specialized.click_sequence.trs_constant import *
from simulator.specialized.click_sequence.datasets import *
from simulator.specialized.click_sequence.data_load_test import *


def custom_loss(predict, label):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    content_loss = criterion(predict, label)
    predict_after = torch.argmax(predict, dim=1)

    lengths_loss = []
    step = 22
    for i in range(0, len(label), step):
        subseq1 = label[i:i + step]
        subseq2 = predict_after[i:i + step]
        non_zero_seq1 = subseq1[(subseq1 != 0)]
        non_zero_seq2 = subseq2[(subseq2 != 0)]
        if 2 in non_zero_seq2:
            index_2 = (non_zero_seq2 == 2).nonzero(as_tuple=False)[0]
            non_zero_seq2 = non_zero_seq2[:index_2]
        length_loss = abs(len(non_zero_seq1) - len(non_zero_seq2))
        lengths_loss.append(length_loss)

    avge_length_acc = sum(lengths_loss)/len(lengths_loss)
    total_loss = 0.2*avge_length_acc + 0.8*content_loss
    return total_loss



if __name__ == "__main__":
    if not os.path.exists(model_path):
       os.makedirs(model_path, exist_ok=True)
    X_train, Y_train, Y_train_uuid, X_test, Y_test, Y_test_uuid, X_coxphm_test, Y_coxphm_test, Y_coxphm_test_uuid = train_test_load(
        train_resample)

    print('Data loading complete.')
    enc_inputs, dec_inputs, dec_outputs,uuids = make_data(X_train,Y_train,Y_train_uuid)
    test_enc_inputs, test_dec_inputs, test_dec_outputs,test_uuids = make_data(X_test,Y_test,Y_test_uuid)

    loader = Data1.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batchsize, True)

    model = Transformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr= lr)

    best_set_acc, best_bleu4 = 0,0
    i = 0
    for epoch in range(epochs):
        model.train()
        for enc_inputs, dec_inputs, dec_outputs in loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                                                            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = custom_loss(outputs, dec_outputs.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        #per epoch test evaluation
        starttime = datetime.datetime.now()
        rouge_2_f,rouge_l_f,rouge_w_f,rouge_s_f,set_acc,bleu4,truedoc_seqlen,virdoc_seqlen = eval_of_train(model, test_enc_inputs, test_dec_outputs,epoch)
        endtime = datetime.datetime.now()
        time_diff = endtime - starttime
        minutes_diff = time_diff.total_seconds() / 60
        print(f'测试集预测用时{minutes_diff}分钟')
        seq_diff = round(abs(truedoc_seqlen - virdoc_seqlen)/truedoc_seqlen,2)

        if set_acc > best_set_acc or bleu4 > best_bleu4:
            if set_acc >= 0.79 and seq_diff < 0.15 :
                torch.save(model.state_dict(), f'{model_path}model_set_acc_{set_acc}_truedoc_seqlen_{truedoc_seqlen}_predict_seqlen_{virdoc_seqlen}.pth')
                print(f'保持模型{model_path}model_set_acc_{set_acc}_truedoc_seqlen_{truedoc_seqlen}_predict_seqlen_{virdoc_seqlen}.pth')
            i = 0
            if set_acc > best_set_acc:
                best_set_acc = set_acc
            if bleu4 > best_bleu4:
                best_bleu4 = bleu4
        else:
            i +=1



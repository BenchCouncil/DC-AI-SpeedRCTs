import torch
import torch.utils.data as Data1

tgt_vocab = {'P': 0, 'S': 1, 'E': 2,'历史基础信息': 3, '既往病史': 4, '历史_血常规': 5, '历史_动脉血气分析': 6, '历史_止凝血': 7, '历史_影像检查': 8, '历史_病原检查': 9,'历史_培养': 10,
             '历史_涂片': 11,'历史用药': 12,'下一步_降钙素原': 13,'下一步_血常规': 14,'下一步_动脉血气分析': 15,'下一步_止凝血': 16,'下一步_影像检查': 17,'下一步_病原检查': 18,'下一步_培养': 19,'下一步_涂片': 20}


def make_data(X,Y,Y_uuid):
    enc_inputs, dec_inputs, dec_outputs,uuids = [], [], [],[]
    for x,y,uuid in zip(X,Y,Y_uuid):
        enc_input = x
        dec_input = y[:-1]
        dec_output = y[1:]
        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)
        uuids.append(uuid)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs),uuids


class MyDataSet(Data1.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

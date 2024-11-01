from project_path import *


#------model parameter---------
d_model = 256   # The dimension of the word Embedding
d_ff = 1024     # Forward propagation hidden layer dimension
d_k = d_v = 64  # K(=Q), V dimensions
n_layers = 6    # Number of encoders and decoders
n_heads = 8     # Multi-Head Attention is 8



batchsize = 64 #when training is 64
lr= 0.0001
epochs = 10
model_path = f'{pro_path}model_save/sepsis_model/transformer/'
test_model_name = 'model_set_acc_0.8_truedoc_seqlen_5.18_predict_seqlen_6.01.pth'


device = 'cuda:0'
tgt_vocab_size = 21     # Target dictionary size
src_vocab_size = 12239  # Number of dictionary words
tgt_len = 21+2  #Maximum length of label

train_resample = 10 #when training is 10
test_resample = 1

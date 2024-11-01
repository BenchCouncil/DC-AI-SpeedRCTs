from transformers import AutoTokenizer, AutoModel
import numpy as np
from project_path import *

biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_path)
biobert_model = AutoModel.from_pretrained(biobert_path)


def get_biobert_embeddings(text):
    tokens_pt = biobert_tokenizer(text, return_tensors="pt")
    outputs = biobert_model(**tokens_pt)
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    hidden_embeddings = last_hidden_state.detach().numpy()
    embeddings = pooler_output.detach().numpy()
    return embeddings, hidden_embeddings


def split_note_document(text, min_length=15):

    tokens_list_0 = biobert_tokenizer.tokenize(text)
    if len(tokens_list_0) <= 510:
        return [text], [1]

    chunk_parse = []
    chunk_length = []
    chunk = text
    ## Go through text and aggregate in groups up to 510 tokens (+ padding)
    tokens_list = biobert_tokenizer.tokenize(chunk)
    if len(tokens_list) >= 510:
        temp = chunk.split('\n')
        ind_start = 0
        len_sub = 0
        for i in range(len(temp)):
            temp_tk = biobert_tokenizer.tokenize(temp[i])
            if len_sub + len(temp_tk) > 510:
                chunk_parse.append(' '.join(temp[ind_start:i]))
                chunk_length.append(len_sub)
                # reset for next chunk
                ind_start = i
                len_sub = len(temp_tk)
            else:
                len_sub += len(temp_tk)
    elif len(tokens_list) >= min_length:
        chunk_parse.append(chunk)
        chunk_length.append(len(tokens_list))
    # print("Parsed lengths: ", chunk_length)

    return chunk_parse, chunk_length


def biobert_embe(full_events_list, event_weights):
    event_weights_exp = []
    for idx, event_string in enumerate(full_events_list):
        weight = event_weights[idx]
        string_list, lengths = split_note_document(event_string)
        for idx_sub, event_string_sub in enumerate(string_list):
            # Extract biobert embedding
            embedding, hidden_embedding = get_biobert_embeddings(event_string_sub)
            # Concatenate
            if (idx == 0) & (idx_sub == 0):
                full_embedding = embedding
            else:
                full_embedding = np.concatenate((full_embedding, embedding), axis=0)
            event_weights_exp.append(weight)

    try:
        aggregated_embedding = np.average(full_embedding, axis=0, weights=np.array(event_weights_exp))
    except:
        aggregated_embedding = np.zeros(768)

    return aggregated_embedding


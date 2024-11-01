from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def pca(dim_num,train_embedding_list):
    train_embedding_df = pd.concat(train_embedding_list, ignore_index=True)
    numeric_columns = train_embedding_df.select_dtypes(include=[np.number]).columns
    sklearn_pca = PCA(n_components=dim_num)
    patient_embedding = sklearn_pca.fit_transform(train_embedding_df[numeric_columns].to_numpy())
    return patient_embedding
from tsfresh import extract_features,feature_extraction
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import numpy as np
from scipy.signal import find_peaks

fc_parameters = {"length": None,
                    "absolute_sum_of_changes": None,
                    "maximum": None,
                    "mean": None,
                    "mean_abs_change": None,
                    "mean_change": None,
                    "median": None,
                    "minimum": None,
                    "standard_deviation": None,
                    "variance": None,
                    "large_standard_deviation": [{"r": r * 0.2} for r in range(1, 5)],
                    "quantile": [{"q": q} for q in [.25, .5, .75, 1]],
                    "linear_trend": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"},{"attr": "slope"}, {"attr": "stderr"}]}


features = {'氯离子', '红细胞-腹液', '部分凝血激酶时间', '红细胞-胸膜', '氧分压', '国际标准化比值', '钾', '酸碱度', '游离钙', '碳氧血红蛋白', '高铁血红蛋白', '红细胞压积-关节液', '淋巴细胞比值', '总二氧化碳', '吸入氧浓度', '尿液中的钠', '血红蛋白浓度', '剩余碱', '肺泡-动脉梯度', '红细胞压积-胸膜', '红细胞压积-腹液', '全血中的钠', '尿素氮', '体温', '葡萄糖', '血氧饱和度', '氧气流量', '碳酸氢根(血清)', '有核红细胞比值', '所需氧气', '呼气末正压', '乳酸', '平均血红蛋白量', '肌酐', '凝血酶原时间', '血小板计数', '嗜碱性粒细胞比值', '红细胞-关节液', '非典型淋巴细胞', '红细胞压积', '红细胞计数', '锂', '中性粒细胞比值', '单核细胞比值', '氧气', 'C反应蛋白', '红细胞平均体积', '纤维蛋白原', '二氧化碳分压', '淋巴细胞绝对值', '嗜酸性粒细胞比值', '平均血红蛋白浓度', '白细胞计数', '红细胞-脑脊液', '体温', '收缩压', '舒张压', '心率', '呼吸频率', '意识', 'qsofa分数'}

def tsfresh_embedding(df):
    for feature in features:
        # 检查 DataFrame 是否包含该特征列
        if feature not in df.columns:
            # 如果不存在，则添加新列并填充为0
            df[feature] = np.nan
    extracted_features = extract_features_haim(df)
    return extracted_features

def extract_features_haim(df_pivot):
    try:
        df_out = df_pivot[['UNIQUE_ID']].iloc[0]
    except:
        df_out = pd.DataFrame(columns=['UNIQUE_ID'])

    # Adding a row of zeros to df_pivot in case there is no value
    new_row = pd.Series(0, index=df_pivot.columns)
    df_pivot = pd.concat([df_pivot, new_row.to_frame().T], ignore_index=True)
    df_pivot['时间'] = df_pivot['时间'].astype(str)
    df_pivot = df_pivot.sort_values(by='时间', ascending=True)
    df_pivot = df_pivot.applymap(lambda x: pd.to_numeric(x, errors='ignore'))

    for feature in features:
        series = df_pivot[feature].dropna() #dropna rows
        if len(series) > 0:
            df_out[feature + '_max'] = series.max()
            df_out[feature + '_min'] = series.min()
            df_out[feature + '_mean'] = series.mean(skipna=True)
            df_out[feature + '_variance'] = series.var(skipna=True)
            df_out[feature + '_meandiff'] = series.diff().mean()  # average change
            df_out[feature + '_meanabsdiff'] = series.diff().abs().mean()
            df_out[feature + '_maxdiff'] = series.diff().abs().max()
            df_out[feature + '_sumabsdiff'] = series.diff().abs().sum()
            df_out[feature + '_diff'] = series.iloc[-1] - series.iloc[0]
            # Compute the n_peaks
            peaks, _ = find_peaks(series)  # , threshold=series.median()
            df_out[feature + '_npeaks'] = len(peaks)
            # Compute the trend (linear slope)
            if len(series) > 1:
                df_out[feature + '_trend'] = np.polyfit(np.arange(len(series)), series, 1)[0]  # fit deg-1 poly
            else:
                df_out[feature + '_trend'] = 0
    del df_out['UNIQUE_ID']
    df_out = df_out.fillna(value=0)
    return df_out


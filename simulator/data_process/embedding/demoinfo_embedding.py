import ast
import pandas as pd
import numpy as np

gender_int = {
    '男':1,
    '女':0
}

mar_status_int = {
    '丧偶':0,
    '单身':1,
    '离异':2,
    '未知':3,
    '已婚':4
}

ethnic_int = {
    '美洲印第安人/阿拉斯加土著人':0,
    '黑人/非洲裔美国人':1,
    '无法获取':2,
    '西班牙裔/拉丁裔':3,
    '白人':4,
    '亚洲人':5,
    '未知':6,
    '其他':7
}

def demoinfo_embedding(row):
    data_dict = ast.literal_eval(row['人口学信息'])
    gender = gender_int.get(data_dict['性别'])
    age = data_dict['年龄']
    ma_status = mar_status_int.get(data_dict['婚姻状态'])
    ethnic = ethnic_int.get(data_dict['种族'])
    return np.array([gender,age,ma_status,ethnic])





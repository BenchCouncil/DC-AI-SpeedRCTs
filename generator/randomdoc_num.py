import sys
from project_path import pro_path
sys.path.append(pro_path)
from simulator.data_process.embedding.doctor_embedding import *
import random
import time


random.seed(123)

def random_value_by_prob(df, column_name):
    column_values = df[column_name]
    value_counts = column_values.value_counts()
    total_count = len(column_values)
    probabilities = value_counts / total_count
    value_prob_dict = probabilities.to_dict()
    values = list(value_prob_dict.keys())
    probabilities = list(value_prob_dict.values())
    chosen_value = random.choices(values, probabilities)[0]
    return chosen_value


def generate_doctor(num,df):
    random_doc_list = []
    title_counts = df['doctor_title'].value_counts()
    title_proba = title_counts / len(title_counts)
    title_prob_dict = title_proba.to_dict()

    title_num_dict = {}
    total_weight = sum(title_prob_dict.values())
    for key, prob in title_prob_dict.items():
        weight = prob / total_weight
        title_num_dict[key] = round(weight * num)

    for title in title_num_dict.keys():
        num_kind = title_num_dict.get(title)

        for i in range(num_kind):
            df_t = df[df['doctor_title'] == title]
            group_unit = random_value_by_prob(df_t, 'doctor_unit')
            group_sex = random_value_by_prob(df_t, 'doctor_sex')
            group_age = random_value_by_prob(df_t, 'doctor_age')
            group_year = random_value_by_prob(df_t, 'doctor_year')
            group_depart = random_value_by_prob(df_t, 'doctor_depart')
            group_field = random_value_by_prob(df_t, 'doctor_field')
            random_doc_list.append((group_unit, group_sex, group_age, group_year, title, group_field, group_depart))
    return random_doc_list

path = f'{pro_path}datasets/csv_and_pkl/data_0321_7000.csv'

df = pd.read_csv(path,encoding='gbk',usecols=['uuid','diag_seq','doctor_unit','doctor_sex','doctor_age','doctor_year','doctor_depart','doctor_title','doctor_field'])

df['doctor_sex'] = df['doctor_sex'].map(doctor_sex_dict)
df['doctor_depart'] = df['doctor_depart'].map(doctor_depart_dict)
df['doctor_title'] = df['doctor_title'].map(doctor_title_dict)
df['doctor_field'] = df['doctor_field'].map(doctor_field_dict)

random_doc_list = generate_doctor(125,df)




def time_of_genrandomdoc(param):
    # Statistics on the time taken to generate virtual doctors
    start_time = time.time()

    generate_doctor(param, df)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(f"programme runtime: {elapsed_minutes:.2f} minutes，that is {elapsed_time}seconds ")


if __name__ == '__main__':
    param = sys.argv[1] # User-defined number of randomised doctors

    time_of_genrandomdoc(param)




import sys
from project_path import pro_path
sys.path.append(pro_path)
import xgboost as xgb
import optuna
import pandas as pd
import csv
from simulator.utils.utils_dataloader import *
from simulator.utils.utils_io_model import *

def train():
    kforder = 0
    for train_index, val_index in skf.split(kflod_em, kflod_label):
        kforder += 1

        X_train, X_val = kflod_em[train_index], kflod_em[val_index]
        y_train, y_val = kflod_label[train_index], kflod_label[val_index]

        model_path = root + f'model_save/{model_sort}_model/{flag}_{model}_dim_{feature_dimnum}_kford_{kforder}_'
        study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))  # maximize  minimize
        study.optimize(lambda trial: objective(trial, X_train,y_train,X_val,y_val,test_em,test_label,model_path,kforder), n_trials=n_trials)  # You can adjust the number of trials
        best_params = study.best_trial.params
        best_value = study.best_value
        print(f"Best Parameters: {best_params}")
        print(f"Best Objective Value: {best_value}")


def predict():
    best_kflod,_,best_model = best_auc_model(root + f'model_save/{model_sort}_model/',flag,model)
    kforder = 1
    for train_index, val_index in skf.split(kflod_em, kflod_label):
        if str(kforder) == best_kflod:
            X_train, X_val = kflod_em[train_index], kflod_em[val_index]
            y_train, y_val = kflod_label[train_index], kflod_label[val_index]

            print('-----Evaluation on the train set-----')
            multu_class_eval(y_train, best_model.predict_proba(X_train))
            print('-----Evaluation on the val set-----')
            multu_class_eval(y_val, best_model.predict_proba(X_val))
            print('-----Evaluation on the test set-----')
            multu_class_eval(test_label, best_model.predict_proba(test_em))
        kforder += 1


def add_next_act():
    root = pro_path
    df = pd.read_csv(root + 'datasets/csv_and_pkl/data_0321_7000.csv', encoding='gbk')
    _, _, best_model = best_auc_model(root + f'model_save/{model_sort}_model/', flag, model)  # 模型名字最后一个acc
    df_new = pd.DataFrame(columns=df.columns)

    df_new_coxphm = pd.DataFrame(columns=df.columns)
    fn_pkl_coxphm = root + f'datasets/{model_sort}_model_input/{flag}_data_7000_dim_{feature_dimnum}_coxphm.pkl'
    data_coxphm = Data(fn=fn_pkl_coxphm, flag=flag)
    for index, row in df.iterrows():
        uuid = row['uuid']  # uuid是唯一标识
        if uuid in data.uuid:
            embedding = data.get_emb_by_uuid(uuid)
            predict_nextact = best_model.predict([embedding])
            row['predict_nextact'] = predict_nextact[0]
            df_new = df_new.append(row, ignore_index=True)
        if uuid in data_coxphm.uuid:
            embedding = data_coxphm.get_emb_by_uuid(uuid)
            predict_nextact = best_model.predict([embedding])
            row['predict_nextact'] = predict_nextact[0]
            df_new_coxphm = df_new_coxphm.append(row, ignore_index=True)
    df_new.to_csv(root + f'datasets/{model_sort}_model_input/data_0321_7000_{model_sort}_nextact.csv', mode='w',
                  index=False, encoding='gbk', header=True, quoting=csv.QUOTE_ALL)
    df_new_coxphm.to_csv(root + f'datasets/{model_sort}_model_input/data_0321_7000_{model_sort}_nextact_coxphm.csv',
                         mode='w', index=False, encoding='gbk', header=True, quoting=csv.QUOTE_ALL)




def objective(trial,train_x,train_y,val_x,val_y,test_x,test_y,model_path,kforder):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators',200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.9, 1.1),
        'max_delta_step': trial.suggest_float('max_delta_step', 0.0, 10.0),  # subsample参数
    }
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        seed=42,
        tree_method='gpu_hist',
        gpu_id=0,
        verbosity=0,
        importance_type= 'gain',
        early_stopping_rounds=50,
        num_class=8,
        **params  # Pass the sampled parameters to the XGBClassifier
    )

    model.fit(train_x,train_y,eval_set=[(val_x,val_y)],verbose=False)
    print(f'----kforder:{kforder}-Evaluation on the train set-----')
    multu_class_eval(train_y, model.predict_proba(train_x))
    print(f'----kforder:{kforder}-Evaluation on the val set-----')
    multu_class_eval(val_y, model.predict_proba(val_x))
    print(f'----kforder:{kforder}--Evaluation on the test set----')
    acc,auc = multu_class_eval(test_y, model.predict_proba(test_x))
    global best_acc
    if acc > best_acc:
        print('save model')
        xgb_save_model(model, model_path=model_path + f'auc_{auc}_acc_{acc}.dat')
        best_acc = acc
    return acc


#-------global parameter-----------
model_sort = 'normal_3h'
flag = 'first'
feature_dimnum = 17
model = 'nextact'
n_trials = 5000
root = pro_path
fn_pkl = root + f'datasets/{model_sort}_model_input/{flag}_data_7000_dim_{feature_dimnum}.pkl'
_,best_acc,_ = best_auc_model(root + f'model_save/{model_sort}_model/',flag,model) #模型名字最后一个acc
#------------------------

data = Data(fn=fn_pkl, flag=flag)
kflod_em, kflod_label, test_em, test_label = data_split(data,flag, model)
skf = KFold(n_splits=5, shuffle=True, random_state=42)
kflod_label = np.array([int(num * 8) for num in kflod_label])
test_label = np.array([int(num * 8) for num in test_label])


if __name__ == '__main__':
    param = sys.argv[1]
    if param == 'train':
        train()
    elif param == 'test':
        predict()
    elif param == 'predict':
        add_next_act()









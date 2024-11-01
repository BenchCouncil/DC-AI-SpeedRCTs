import sys
from project_path import pro_path
sys.path.append(pro_path)
import xgboost as xgb
import optuna
import pandas as pd
from simulator.utils.utils_dataloader import *
from simulator.utils.utils_io_model import *

def train():
    kforder = 1
    for train_index, val_index in skf.split(kflod_em, kflod_label):
        X_train, X_val = kflod_em[train_index], kflod_em[val_index]
        y_train, y_val = kflod_label[train_index], kflod_label[val_index]

        model_path = root + f'model_save/{model_sort}_model/{flag}_{model}_dim_{feature_dimnum}_kford_{kforder}_'
        study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42))  # maximize
        study.optimize(lambda trial: objective(trial, X_train,y_train,X_val,y_val,test_em,test_label,model_path,kforder), n_trials=n_trials)  # You can adjust the number of trials
        kforder += 1
        best_params = study.best_trial.params
        best_value = study.best_value
        print(f"Best Parameters: {best_params}")
        print(f"Best Objective Value: {best_value}")


def predict():
    best_kflod,lowest_rmse,best_model = lowest_rmse_model(root + f'model_save/{model_sort}_model/',flag,model)
    kforder = 1
    for train_index, val_index in skf.split(kflod_em, kflod_label):
        if str(kforder) == best_kflod:
            X_train, X_val = kflod_em[train_index], kflod_em[val_index]
            y_train, y_val = kflod_label[train_index], kflod_label[val_index]

            print(f'-----kforder:{kforder}--Evaluation on the train set-----')
            rmse_eval(y_train, best_model.predict(X_train))
            print(f'-----kforder:{kforder}--Evaluation on the val set-----')
            rmse_eval(y_val, best_model.predict(X_val))
            print(f'-----kforder:{kforder}--Evaluation on the test set-----')
            rmse_eval(test_label, best_model.predict(test_em))
            print(f'-----kforder:{kforder}--Evaluation on the test set(Label is range)-----')
            test_y_range = diagtime_label_to_range(test_label, best_model.predict(test_em), 0.2)
            _, mae_after = rmse_eval(test_y_range, best_model.predict(test_em))

        kforder += 1



def objective(trial,train_x,train_y,val_x,val_y,test_x,test_y,model_path,kforder):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'n_estimators': trial.suggest_int('n_estimators',200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
    }
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        seed=42,
        tree_method='gpu_hist',
        gpu_id=0,
        verbosity=0,
        importance_type= 'gain',
        **params  # Pass the sampled parameters to the XGBClassifier
    )

    model.fit(train_x,train_y,eval_set=[(val_x,val_y)],early_stopping_rounds=50,verbose=False)
    print(f'----kforder:{kforder}-Evaluation on the train set-----')
    rmse_eval(train_y,model.predict(train_x))
    print(f'----kforder:{kforder}-Evaluation on the val set-----')
    rmse_eval(val_y, model.predict(val_x))
    print(f'----kforder:{kforder}--Evaluation on the test set----')
    rmse, mae = rmse_eval(test_y,model.predict(test_x))
    print(f'-----kforder:{kforder}--Evaluation on the test set(Label is range)-----')
    test_y_range = diagtime_label_to_range(test_y, model.predict(test_em), 0.2)
    _,mae_after = rmse_eval(test_y_range, model.predict(test_em))
    global lowest_mae
    if mae < lowest_mae:
        print('save model')
        xgb_save_model(model, model_path=model_path + f'mae_{mae_after}.dat')
        lowest_mae = mae
    return mae





#-------global parameter-----------
model_sort = 'normal_0h'
flag = 'final'
feature_dimnum = 18
model = 'diagtime'
n_trials = 10000
root = pro_path
fn_pkl = root + f'datasets/{model_sort}_model_input/{flag}_data_7000_dim_{feature_dimnum}.pkl'
_, lowest_mae, _ = lowest_rmse_model(root + f'model_save/{model_sort}_model/', flag, model)
#------------------------

data = Data(fn=fn_pkl, flag=flag)
kflod_em, kflod_label, test_em, test_label = data_split(data,flag, model)
skf = KFold(n_splits=5, shuffle=True, random_state=42)


if __name__ == '__main__':
    param = sys.argv[1]
    if param == 'train':
        train()
    elif param == 'test':
        predict()











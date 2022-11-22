import olorenchemengine as oce
import numpy as np
import pandas as pd
from tqdm import tqdm
from tdc import Evaluator
from tdc.benchmark_group import admet_group
    

def model_from_dict(s):
    s = s.replace("'", "\"").replace("True", "true").replace("False", "false").replace("None", "null")
    model = oce.create_BC(s)
    return model

def train_bbb_baselines():
    """ 
        Returns tuple of 4 trained BBB models on TDC data.
    """
    TASK_NAME = 'bbb_martins'
    models = {
    'BBB_TDC': '{"BC_class_name": "BaseBoosting", "args": [[{"BC_class_name": "RandomForestModel", "args": [{"BC_class_name": "DescriptastorusDescriptor", "args": ["rdkit2dnormalized"], "kwargs": {}}], "kwargs": {"n_estimators": 1000, "max_features": "log2", "max_depth": null, "criterion": "entropy", "class_weight": null, "bootstrap": true}}, {"BC_class_name": "RandomForestModel", "args": [{"BC_class_name": "DescriptastorusDescriptor", "args": ["morgan3counts"], "kwargs": {}}], "kwargs": {"n_estimators": 1000, "max_features": "log2", "max_depth": null, "criterion": "entropy", "class_weight": null, "bootstrap": true}}, {"BC_class_name": "RandomForestModel", "args": [{"BC_class_name": "OlorenCheckpoint", "args": ["default"], "kwargs": {"num_tasks": 2048}}], "kwargs": {"n_estimators": 1000, "max_features": "log2", "max_depth": null, "criterion": "entropy", "class_weight": null, "bootstrap": true}}]], "kwargs": {"n": 1, "log": true}}',
    'BBB_1040':"{'BC_class_name': 'BaseBoosting', 'args': [[{'BC_class_name': 'RandomForestModel', 'args': [{'BC_class_name': 'DescriptastorusDescriptor', 'args': ['rdkit2dnormalized'], 'kwargs': {'log': True, 'scale': None}}], 'kwargs': {'n_estimators': 1000, 'max_features': 'log2', 'max_depth': None, 'criterion': 'entropy', 'class_weight': None, 'bootstrap': True}}, {'BC_class_name': 'RandomForestModel', 'args': [{'BC_class_name': 'DescriptastorusDescriptor', 'args': ['morgan3counts'], 'kwargs': {'log': True, 'scale': None}}], 'kwargs': {'n_estimators': 1000, 'max_features': 'log2', 'max_depth': None, 'criterion': 'entropy', 'class_weight': None, 'bootstrap': True}}, {'BC_class_name': 'RandomForestModel', 'args': [{'BC_class_name': 'OlorenCheckpoint', 'args': ['default'], 'kwargs': {'num_tasks': 2048, 'log': True}}], 'kwargs': {'n_estimators': 1000, 'max_features': 'log2', 'max_depth': None, 'criterion': 'entropy', 'class_weight': None, 'bootstrap': True}}]], 'kwargs': {'n': 1, 'log': True, 'oof': False, 'nfolds': 5}}",
    'BBB_455': "{'BC_class_name': 'RandomForestModel', 'args': [{'BC_class_name': 'DescriptastorusDescriptor', 'args': ['rdkit2dnormalized'], 'kwargs': {'log': True, 'scale': None}}], 'kwargs': {'n_estimators': 2000, 'max_features': 'log2', 'max_depth': None, 'criterion': 'entropy', 'class_weight': None, 'bootstrap': True}}",
    'BBB_1209':"{'BC_class_name': 'ChemPropModel', 'args': [], 'kwargs': {'epochs': 100, 'dropout_rate': 0.0, 'batch_size': 50, 'lr': 0.001, 'hidden_size': 300, 'depth': 3}}"
    }

    print("____TRAINING BASELINES_____")
    model_dict = {}
    for MODEL in models:
        group = admet_group(path = 'data/')
        benchmark = group.get(TASK_NAME) 
        train = benchmark['train_val']
        model = model_from_dict(models[MODEL])
        model.fit(train['Drug'], train['Y'])
        model_dict[MODEL] = model
    print("____TRAINING COMPLETED_____")

    return model_dict


def baselines_vs_b3db():
    model_dict = train_bbb_baselines()

    results_dict = {
                    'DATASET':  [],
                    'BBB_TDC':  [],
                    'BBB_1040': [],
                    'BBB_455':  [],
                    'BBB_1209': [],
    }
    
    DATASET_LIST = ['R6',  'R7',  'R9',  'R10', 'R13', 
                    'R14', 'R15', 'R16', 'R19', 'R23', 
                    'R24', 'R26', 'R27', 'R28', 'R29', 
                    'R30', 'R36', 'R37', 'R50']

    DATA_FOLDER= 'b3db_class_data'
    for DATASET in DATASET_LIST:
        results_dict['DATASET'].append(DATASET)
        curr_dataset = pd.read_csv(f"{DATA_FOLDER}/{DATASET}/{DATASET}_cleaned.csv")
        print(f"___________________________")
        for MODEL in model_dict.keys():
            print(MODEL, DATASET)
            y_pred_test = model_dict[MODEL].predict(curr_dataset['Drug'])
            calculate_auroc = Evaluator(name = 'ROC-AUC')
            AUROC = calculate_auroc(curr_dataset["Y"], y_pred_test)
            AUROC = round(AUROC, 3)
            results_dict[MODEL].append(AUROC)
            print(results_dict[MODEL])
    res_table = pd.DataFrame(results_dict)
    res_table.to_csv(f"{DATA_FOLDER}/generalizability_pred.csv")

if __name__ == '__main__':
    baselines_vs_b3db()



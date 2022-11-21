import olorenchemengine as oce
import numpy as np
import pandas as pd
from tqdm import tqdm
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
        completed = None
        for seed in tqdm([1, 2, 3, 4, 5]):
            benchmark = group.get(TASK_NAME) 
            train = benchmark['train_val']
            model = model_from_dict(MODEL)
            model.fit(train['Drug'], train['Y'])
            completed = model
        model_dict[MODEL] = completed
    print("____TRAINING COMPLETED_____")
    return model_dict

    

def baselines_vs_b3db():
    model_dict = train_bbb_baselines()
    TASK_NAME = 'bbb_martins'
    
    results_dict = {'DATASET':        [],
                    'BBB_TDC AUROC':  [],
                    'BBB_TDC Error':  [],
                    'BBB_1040 AUROC': [],
                    'BBB_1040 Error': [],
                    'BBB_455 AUROC':  [],
                    'BBB_455 Error':  [],
                    'BBB_1209 AUROC': [],
                    'BBB_1209 Error': [],
    }
    res_table = pd.DataFrame(results_dict)
    DATASET_LIST = ['R6', 'R7', 'R9', 'R10', 'R13', 
                    'R14', 'R15', 'R16', 'R19', 'R23', 
                    'R24', 'R26', 'R27', 'R28', 'R29', 
                    'R30', 'R36', 'R37', 'R50']
    DATA_PATH = 'b3db_class_data/'
    for model in model_dict.keys():
        print(f"__________________________")
        group = admet_group(path = 'data/')
        for DATASET in DATASET_LIST:
            if DATASET == 'R7': break
            benchmark = group.get(TASK_NAME) 
            predictions = {}
            name = benchmark['name']
            curr_dataset = pd.read_csv(f"{DATA_PATH}{DATASET}/{DATASET}_cleaned.csv")
            predictions_list = []
            for seed in tqdm([1, 2, 3, 4, 5]):
                y_pred_test = model.predict(curr_dataset['Drug'])
                predictions[name] = y_pred_test
                predictions_list.append(predictions)
            results = group.evaluate_many(predictions_list)

            AUROC = results[0]
            ERROR = results[1]
            if model == 'BBB_TDC':
                res_table.append({'DATASET': DATASET,
                                  'BBB_TDC AUROC': AUROC,
                                  'BBB_TDC Error': ERROR})
            if model == 'BBB_1040':
                res_table.append({'DATASET': DATASET,
                                  'BBB_1040 AUROC': AUROC,
                                  'BBB_1040 Error': ERROR})
            if model == 'BBB_455':
                res_table.append({'DATASET': DATASET,
                                  'BBB_455 AUROC': AUROC,
                                  'BBB_455 Error': ERROR})
            if model == 'BBB_1209':
                res_table.append({'DATASET': DATASET,
                                  'BBB_1209 AUROC': AUROC,
                                  'BBB_1209 Error': ERROR})
    res_table.to_csv(f"{DATA_PATH}generalizability_pred.csv")

if __name__ == '__main__':
    baselines_vs_b3db()



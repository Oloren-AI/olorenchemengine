import olorenchemengine as oce
import numpy as np
import pandas as np
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
        print(f"__________________________")
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



    results_dict = {}
    for MODEL in model_dict.keys():
        print(f"__________________________")
        group = admet_group(path = 'data/')
        predictions_list = []
        for seed in tqdm([1, 2, 3, 4, 5]):
            benchmark = group.get(TASK_NAME) 
            predictions = {}
            name = benchmark['name']
            train, test = benchmark['train_val'], benchmark['test']
            model = model_from_dict(model_dict[MODEL])
            model.fit(train['Drug'], train['Y'])
            y_pred_test = model.predict(test['Drug'])
            oce.save(model, MODEL)
            predictions[name] = y_pred_test
            predictions_list.append(predictions)
        results = group.evaluate_many(predictions_list)
        results_dict[MODEL] = results[TASK_NAME]
    print(results_dict)

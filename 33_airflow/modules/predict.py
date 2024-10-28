import json
import os
import dill
import logging
import pandas as pd



def get_jsons_from_dir(path)->list:
    jsons = []
    for x in os.listdir(path):
        with open(path + x, 'rb') as f:
            jsons.append(json.load(f))

    return jsons


def get_last_model_num(path):
    models_nums = [file_name.split('_')[2][:-4] for file_name in os.listdir(path)]
    sorted(models_nums, key=lambda x: int(x))
    last_model_num = models_nums[-1]

    return last_model_num



def predict():
    path = os.environ['PROJECT_PATH']

    models_path = f'{path}/data/models/'
    last_model_num = get_last_model_num(models_path)

    with open(f'{models_path}/cars_pipe_{last_model_num}.pkl', 'rb') as f:
        model_pipline = dill.load(f)

    test_path = f'{path}/data/test/'
    jsons = get_jsons_from_dir(test_path)
    x_test = pd.DataFrame(jsons)

    predictions = model_pipline.predict(x_test)
    predictions_df = [[car_id, predict] for car_id, predict in zip(x_test['id'], predictions)]
    predictions_df = pd.DataFrame(predictions_df, columns=['car_id', 'predict'])

    predictions_df.to_csv(f'{path}/data/predictions/predictions_{last_model_num}.csv')

if __name__ == '__main__':
    predict()

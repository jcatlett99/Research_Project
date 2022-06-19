import regression
import formatting as form
import pandas as pd
import numpy as np
import os


def fix_dataset():
    chosen_values = form.read_csv()
    for item in os.listdir('moshuir_data/datasets'):
        if not item.startswith('.'):
            filename = os.fsdecode(item)
            if filename not in chosen_values:
                os.rename("moshuir_data/datasets/"+filename, "moshuir_data/not_used/"+filename)


def temp(data_frame):

    chosen_values = form.read_csv()
    data_frame['aesthetic'] = np.zeros(len(data_frame))
    for k in chosen_values:
        data_frame.at[k, 'aesthetic'] = chosen_values[k]

    data_frame.to_csv('moshuir_data/table.csv', encoding='utf-8')

    r = regression.Regression
    features = ['histograms',
                'entropy',
                'straight_diagonal_line_ratio',
                'horizontal_vertical_line_ration',
                'diagonal_dominance',
                'symmetry',
                'rule_of_thirds_power_points',
                'rule_of_thirds_gridlines',
                'sharpness',
                'contrast',
                'luminance',
                'saturation']
    r.linear_regression(features)
    # variables = ['novelty', 'luminance', 'rule_of_thirds_gridlines', 'saturation']
    # r.verify_model(variables)
    # coeff = []
    # r.run_model(coeff)


temp(pd.read_csv('output/table.csv', index_col=0))
# fix_dataset()

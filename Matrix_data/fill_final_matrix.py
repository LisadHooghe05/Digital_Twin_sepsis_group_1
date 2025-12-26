import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def fill_matrix_with_zeros():
    final_matrix = pd.read_csv('matrix_overview_without_all_NaN.csv')
    final_matrix = final_matrix.replace(r"^\s*$", np.nan, regex=True)

    #print("BEFORE drop:", final_matrix.shape)

    #We halen temperature eruit
    final_matrix.drop(columns=['Temperature'], inplace=True)
    # final_matrix.drop(columns=['Temperature', 'Norepinephrine_x'], inplace=True)


    vitals = ['age_12h_before_AKI', 'Diastolic Blood Pressure','Heart Rate',
              'Mean Arterial Pressure','Oxygen Saturation','Respiratory Rate',
               'Systolic Blood Pressure'
               ]
    
    other_features = []
    medians_used = {}
    for column in final_matrix.columns:
        if column not in vitals:
            other_features.append(column)
    
    for vital in vitals:
        median = final_matrix[vital].median()
        medians_used[vital] = median
        final_matrix[vital] = final_matrix[vital].fillna(median)
    

    for feature in other_features:
        final_matrix[feature] = final_matrix[feature].fillna(0)

    missing_values = pd.DataFrame({
        'feature': final_matrix.columns,
        'missing_values': final_matrix.isna().sum().values
    }) 

    medians_df = pd.DataFrame({
        "feature": list(medians_used.keys()),
        "median": list(medians_used.values())
    })

    #print("AFTER drop:", final_matrix.shape)

    #final_matrix.to_csv('matrix_final_final.csv', index=False)

    return missing_values, medians_df,final_matrix

# result_1, result_2 = fill_matrix_with_zeros()
# print(result_1[result_1['missing_values'] > 0])
# print(result_2)




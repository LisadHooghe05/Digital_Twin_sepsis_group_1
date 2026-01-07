import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def fill_matrix_with_zeros():
    """
    Is a function that fills the file matrix_overview_without_all_NaN with zero's if there was a NaN in the matrix
    Because some clustering methods do not like NaN values.

    Returns:
    - final_matrix: a pd dataframe that will be used for the clustering
    - missing values: shows how much values are still missing in the matrix
    - medians: gives values for the medians that are used to fill the matrix by vitals, because
      someone needs to have vitals
    """
    final_matrix = pd.read_csv('matrix_overview_without_all_NaN.csv')
    final_matrix = final_matrix.replace(r"^\s*$", np.nan, regex=True)

    # print("BEFORE drop:", final_matrix.shape)

    # Do not use the column temperature
    final_matrix.drop(columns=['Temperature'], inplace=True)

    vitals = ['age_12h_before_AKI', 'Diastolic Blood Pressure','Heart Rate',
              'Mean Arterial Pressure','Oxygen Saturation','Respiratory Rate',
               'Systolic Blood Pressure']
    
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
        'missing_values': final_matrix.isna().sum().values}) 

    medians_df = pd.DataFrame({
        "feature": list(medians_used.keys()),
        "median": list(medians_used.values())})

    # print("AFTER drop:", final_matrix.shape)

    # final_matrix.to_csv('matrix_final_final.csv', index=False)

    return missing_values, medians_df,final_matrix

if __name__ == "__main__":
    result_1, result_2, result_3 = fill_matrix_with_zeros()
    # print(result_1[result_1['missing_values'] > 0])
    print(result_2)
    # print(result_3)






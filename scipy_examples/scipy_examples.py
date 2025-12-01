from scipy.sparse import csr_matrix, hstack
import numpy as np
import pandas as pd

divider = "========== ========== ========== ========== =========="


def create_dataframe():
    data = np.array([[0, 1, 2], [5, 3, 8], [8, 6, 5]])
    df = pd.DataFrame(data, columns=["feature_1", "feature_2", "feature_3"])
    return df


mat_1 = np.array([[1, 2, 3], [4, 3, 2], [9, 8, 8]])
mat_2 = np.array([[1, 0, 0], [0, 3, 0]])

mat_1_sparsed = csr_matrix(mat_1)
mat_2_sparsed = csr_matrix(mat_2)
print(mat_1_sparsed)

# print(mat_1_sparsed)
# print(divider)
# print(mat_2_sparsed)
"""
csr_matrix stores only non zeros
using csr_matrix is useful when there are many 0's in data. 
"""
df = create_dataframe()

print(divider)
print(df)

df_sparsed = csr_matrix(df)

print(divider)
# scipy hstack only works with sparsed matrix
combine = hstack([mat_1_sparsed, df_sparsed])
print(combine)

import numpy as np

def gaussNaive(A, B):
    aug = np.column_stack((A, B))
    m, n = A.shape
    nb = aug.shape[1]  # nb = n + 1
    print(aug)

    # forward elimination
    for k in range(m):
        for i in range(k + 1, m):
            factor = aug[i][k] / aug[k][k]
            for j in range(k, nb):
                aug[i][j] -= aug[k][j] * factor
                aug[i][k] = 0
    print(aug)

    # backword substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i, n):
            s += (aug[i][j] * x[j])
        x[i] = (aug[i][n] - s) / aug[i][i]

    return x

def pivot(aug, m, i):
    max_val = abs(aug[i][i])
    max_row = i
    for r in range(i + 1, m):
        if max_val < abs(aug[r][i]):
            max_row = r
            max_val = abs(aug[r][i])
    # 1
    if max_row != i:
        swapRows(aug, i, max_row)

def swapRows(matrix, rowA, rowB):
    temp = matrix[rowA, :].copy()
    matrix[rowA, :] = matrix[rowA, :].copy()
    matrix[rowB, :]

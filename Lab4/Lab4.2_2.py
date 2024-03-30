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

A = np.array([[1,2,1],[2,2,3],[-1,-3,0]], np.float32)
B = np.array([0,3,2],np.float32)
X = gaussNaive(A, B)
result = np.dot(A, X)

# 2
if B.shape[0] != A.shape[0] or not np.array_equal(result, B):
    print("จำนวน Row Matrix B ไม่เท่ากับจำนวน Row ของ Matrix A")
    print("Matrix A ไม่ใช่ Matrix จตุรัส")
else:
    for i, x in enumerate(X):
        print('X%d = %.2f' % (i+1, x))

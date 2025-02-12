# DIMS
N^(L) = N^(L−1)
H^(L) = 1 + (H^(L-1) - filter_y^(L)) / stride_y^(L)
W^(L) = 1 + (W^(L-1) - filter_x^(L)) / stride_x^(L)
C^(L) = C^(L) * filter_y^(L) * filter_x^(L)

# GMM
IN      N/28/28/1
FOLD    X=28 Y=28 S=1 / X=14 Y=14 S=2 / X=12 Y=12 S=2 / X=8 Y=8 S=2
-> (N/1/1/784) / (N/8/8/196) / (N/9/9/144) / (N/11/11/64)
GMM     K=16/25/49/64/81/100
-> (N/1/1/K) / (N/8/8/K) / (N/9/9/K) / (N/11/11/K)
LIN
-> (N/1/1/10)

# DCGMM_2L
IN      N/28/28/1
FOLD    X=8  Y=8  S=2  (N/11/11/64)
GMM     K=25           (N/11/11/25)
FOLD    X=11 Y=11 S=1  (N/1/1/3025)
GMM     K=25           (N/1/1/25)
LIN                    (N/1/1/10)

# DCGMM_3L 
IN      N/28/28/1
FOLD    X=3  Y=3  S=1 (N/26/26/9)
GMM     K=25          (N/26/26/25)
FOLD    X=4  Y=4  S=2 (N/12/12/400)
GMM     K=25          (N/12/12/25)
FOLD    X=12 Y=12 S=1 (N/1/1/3600)
GMM     K=49          (N/1/1/25)
LIN                   (N/1/1/10)
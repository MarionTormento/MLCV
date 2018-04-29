import numpy as np
# RotMap2 and 3
# Auto Harris RGB
# Rotation = 
# [[ 0.98223029 -0.18767968  0.        ]
#  [ 0.18767968  0.98223029  0.        ]
#  [ 0.          0.          1.        ]]
# 0.9999999999999997 0.9999999999999998
# 10.817404059670668
# Translation = 
# [ 36.36791358 -42.60094205   0.        ]
Hauto = [[ 1.34011584, -0.235230943, -47.8576319],
 [ 0.292330265,  1.27674953, -124.839282],
 [ 0.00194898905, 0.0000512082653, 1.00000000]]

Hauto=np.asarray(Hauto)
detAuto = np.linalg.det(Hauto)

# Manual Harris RGB
# Rotation = 
# [[ 0.97561067 -0.21950812  0.        ]
#  [ 0.21950812  0.97561067  0.        ]
#  [ 0.          0.          1.        ]]
# 0.9999999999999996 0.9999999999999998
# 12.680143900906797
# Translation = 
# [ 29.05972091 -50.04782837   0.        ]
Hmanual = [[ 1.45465078, -0.412282798, -55.3734161],
 [ 0.305727105,  1.20259678, -115.423854],
 [ 0.000383956831, -0.000334317678,  1.00000000]]

Hmanual = np.asarray(Hmanual)
detMan = np.linalg.det(Hmanual)
print(detMan)
print(100*abs((Hauto-Hmanual)/Hauto))
print((detAuto-detMan)/detAuto)


from .algebras.exceptionals import F4


algebra = F4()
f1 = algebra.funamdental_weights[0]
algebra.tensor_product_decomp(f1, f1)
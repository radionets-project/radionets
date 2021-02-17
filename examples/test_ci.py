import numpy as np
from radionets.dl_framework.data import load_data

def test_ci():
    a = np.array([1, 2, 3, 4])
    assert len(a) == 4
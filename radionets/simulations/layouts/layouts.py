import numpy as np
from pathlib import Path


file_dir = Path(__file__).parent.resolve()


def vlba():
    x, y, z, _, _ = np.genfromtxt(file_dir / "vlba.txt", unpack=True)
    ant_pos = np.array([x, y, z])
    return ant_pos

def eht():
    _, x, y, z, _, _, _, _, _ = np.genfromtxt(file_dir / "eht.txt", unpack=True)
    print(x)
    ant_pos = np.array([x, y, z])
    return ant_pos

from pathlib import Path

import numpy as np

file_dir = Path(__file__).parent.resolve()


def vlba():
    x, y, z, _, _ = np.genfromtxt(file_dir / "vlba.txt", unpack=True)
    ant_pos = np.array([x, y, z])
    return ant_pos

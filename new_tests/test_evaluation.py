import pytest


@pytest.mark.last
def test_get_images():
    import torch
    from radionets.dl_framework.data import load_data, do_normalisation
    import pandas as pd

    test_ds = load_data(
        "./new_tests/build/data",
        mode="test",
        fourier=True,
        source_list=False,
    )

    num_images = 10
    rand = True
    norm_path = "none"

    indices = torch.arange(num_images)
    assert len(indices) == 10

    if rand:
        indices = torch.randint(0, len(test_ds), size=(num_images,))
    img_test = test_ds[indices][0]

    assert img_test.shape == (10, 2, 63, 63)
    norm = "none"
    if norm_path != "none":
        norm = pd.read_csv(norm_path)
    img_test = do_normalisation(img_test, norm)
    img_true = test_ds[indices][1]

    assert img_true.shape == (10, 2, 63, 63)

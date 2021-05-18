import pytest


@pytest.mark.order("last")
class TestEvaluation:
    def test_get_images(self):
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

    def test_get_prediction(self):
        from radionets.dl_framework.data import load_data
        from radionets.evaluation.utils import (
            get_images,
            load_pretrained_model,
            read_config,
            eval_model,
        )
        import toml
        import torch

        config = toml.load("./new_tests/evaluate.toml")
        conf = read_config(config)

        test_ds = load_data(
            "./new_tests/build/data",
            mode="test",
            fourier=True,
            source_list=False,
        )

        num_images = 10
        rand = True

        if num_images is None:
            num_images = len(test_ds)
        img_test, img_true = get_images(test_ds, num_images, norm_path="none", rand=rand)
        img_size = img_test.shape[-1]
        model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
        pred = eval_model(img_test, model)
        assert str(pred.device) == "cpu"

        # test for uncertainty
        if pred.shape[1] == 4:
            pred_1 = pred[:, 0, :].unsqueeze(1)
            pred_2 = pred[:, 2, :].unsqueeze(1)
            pred = torch.cat((pred_1, pred_2), dim=1)

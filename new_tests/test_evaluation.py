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
        from pathlib import Path
        from radionets.evaluation.utils import (
            read_config,
            save_pred,
        )
        import toml
        import torch

        config = toml.load("./new_tests/evaluate.toml")
        conf = read_config(config)

        pred, img_test, img_true = generate_images(
            conf["arch_name"], conf["model_path"]
        )
        assert str(pred.device) == "cpu"

        # test for uncertainty
        if pred.shape[1] == 4:
            pred_1 = pred[:, 0, :].unsqueeze(1)
            pred_2 = pred[:, 2, :].unsqueeze(1)
            pred = torch.cat((pred_1, pred_2), dim=1)

        assert pred.shape == (10, 2, 63, 63)

        pred = pred.numpy()
        out_path = Path("./new_tests/build/test_training/evaluation/")
        out_path.mkdir(parents=True, exist_ok=True)
        save_pred(
            str(out_path) + "/predictions_model_test.h5",
            pred,
            img_test,
            img_true,
            "pred",
            "img_test",
            "img_true",
        )

    def test_contour(self):
        from radionets.evaluation.utils import (
            read_config,
            get_ifft,
            save_pred,
        )
        from radionets.evaluation.contour import (
            area_of_contour,
        )
        import toml
        import numpy as np

        config = toml.load("./new_tests/evaluate.toml")
        conf = read_config(config)

        pred, img_test, img_true = generate_images(
            conf["arch_name"],
            "./new_tests/model/model_eval.model",
        )

        assert str(pred.device) == "cpu"

        ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])
        ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])

        assert ~np.isnan([ifft_pred, ifft_truth]).any()

        val = area_of_contour(ifft_pred[0], ifft_truth[0])

        assert isinstance(val, np.float64)
        assert ~np.isnan(val).any()
        assert val > 0

        save_pred(
            "./new_tests/build/test_training/evaluation/predictions_model_eval.h5",
            pred,
            img_test,
            img_true,
            "pred",
            "img_test",
            "img_true",
        )

    def test_evaluation(self):
        import shutil
        import os
        from click.testing import CliRunner
        from radionets.evaluation.scripts.start_evaluation import main

        runner = CliRunner()
        result = runner.invoke(main, "new_tests/evaluate.toml")
        assert result.exit_code == 0

        if os.path.exists("new_tests/model/evaluation"):
            shutil.rmtree("new_tests/model/evaluation")


def generate_images(arch_name, model_path):
    from radionets.dl_framework.data import load_data
    from radionets.evaluation.utils import get_images, eval_model, load_pretrained_model

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
    model = load_pretrained_model(arch_name, model_path, img_size)
    pred = eval_model(img_test, model, test=True)

    return pred, img_test, img_true

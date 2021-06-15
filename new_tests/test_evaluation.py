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

        pred, img_test, img_true = generate_images(conf)
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

        pred, img_test, img_true = generate_images(conf)

        assert str(pred.device) == "cpu"

        ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])
        ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])

        assert ~np.isnan([ifft_pred, ifft_truth]).any()

        assert ifft_pred[0].shape == (63, 63)
        assert ifft_truth[0].shape == (63, 63)

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

    def test_im_to_array_value(self):
        import torch
        from radionets.evaluation.utils import read_pred

        pred, img_test, img_true = read_pred(
            "./new_tests/build/test_training/evaluation/predictions_model_eval.h5"
        )

        image = pred[0]

        num = image.shape[0]
        pix = image.shape[-1]

        a = torch.arange(0, pix, 1)
        grid_x, grid_y = torch.meshgrid(a, a)
        x_coords = torch.cat(num * [grid_x.flatten().unsqueeze(0)])
        y_coords = torch.cat(num * [grid_y.flatten().unsqueeze(0)])
        value = image.reshape(-1, pix ** 2)

        assert x_coords.shape == (2, 3969)
        assert y_coords.shape == (2, 3969)
        assert value.shape == (2, 3969)

    def test_bmul(self):
        import torch

        vec = torch.ones(1)
        mat = torch.ones(1, 2, 2)
        axis = 0

        mat = mat.transpose(axis, -1)
        cov = (mat * vec.expand_as(mat)).transpose(axis, -1)

        assert cov.shape == (1, 2, 2)

    def test_pca(self):
        import torch
        import toml
        from radionets.evaluation.jet_angle import im_to_array_value, bmul
        from radionets.evaluation.utils import read_pred, get_ifft, read_config
        from math import pi

        config = toml.load("./new_tests/evaluate.toml")
        conf = read_config(config)

        torch.set_printoptions(precision=16)

        pred, img_test, img_true = read_pred(
            "./new_tests/build/test_training/evaluation/predictions_model_eval.h5"
        )

        ifft_pred = get_ifft(pred, conf["amp_phase"])
        assert ifft_pred.shape == (10, 63, 63)

        pix_x, pix_y, image = im_to_array_value(torch.tensor(ifft_pred))

        cog_x = (torch.sum(pix_x * image, axis=1) / torch.sum(image, axis=1)).unsqueeze(
            -1
        )
        cog_y = (torch.sum(pix_y * image, axis=1) / torch.sum(image, axis=1)).unsqueeze(
            -1
        )

        assert cog_x.shape == (10, 1)
        assert cog_y.shape == (10, 1)

        delta_x = pix_x - cog_x
        delta_y = pix_y - cog_y

        inp = torch.cat([delta_x.unsqueeze(1), delta_y.unsqueeze(1)], dim=1)

        cov_w = bmul(
            (
                cog_x - 1 * torch.sum(image * image, axis=1).unsqueeze(-1) / cog_x
            ).squeeze(1),
            (torch.matmul(image.unsqueeze(1) * inp, inp.transpose(1, 2))),
        )

        eig_vals_torch, eig_vecs_torch = torch.symeig(cov_w, eigenvectors=True)

        assert eig_vals_torch.shape == (10, 2)
        assert eig_vecs_torch.shape == (10, 2, 2)

        psi_torch = (
            torch.atan(eig_vecs_torch[:, 1, 1] / eig_vecs_torch[:, 0, 1]).numpy()
            * 180
            / pi
        )

        assert len(psi_torch) == 10
        assert len(psi_torch[psi_torch > 360]) == 0

    def test_calc_jet_angle(self):
        import torch
        import toml
        from radionets.evaluation.jet_angle import pca
        from math import pi
        from radionets.evaluation.utils import read_config, read_pred, get_ifft

        config = toml.load("./new_tests/evaluate.toml")
        conf = read_config(config)

        torch.set_printoptions(precision=16)

        pred, img_test, img_true = read_pred(
            "./new_tests/build/test_training/evaluation/predictions_model_eval.h5"
        )

        image = get_ifft(pred, conf["amp_phase"])
        assert image.shape == (10, 63, 63)

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        image = image.clone()
        img_size = image.shape[-1]
        # ignore negagive pixels, which can appear in predictions
        image[image < 0] = 0

        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        bs = image.shape[0]

        # only use brightest pixel
        max_val = torch.tensor([(i.max() * 0.4) for i in image])
        max_arr = (torch.ones(img_size, img_size, bs) * max_val).permute(2, 0, 1)
        image[image < max_arr] = 0

        assert image.shape == (10, 63, 63)

        _, _, alpha_pca = pca(image)

        x_mid = torch.ones(img_size, img_size).shape[0] // 2
        y_mid = torch.ones(img_size, img_size).shape[1] // 2

        assert x_mid == 31
        assert y_mid == 31

        m = torch.tan(pi / 2 - alpha_pca)
        n = torch.tensor(y_mid) - m * torch.tensor(x_mid)
        alpha = ((alpha_pca) * 180 / pi).numpy()

        assert len(n) == 10
        assert len(alpha) == 10
        assert len(alpha[alpha > 360]) == 0

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


def generate_images(conf):
    from radionets.dl_framework.data import load_data
    from radionets.evaluation.utils import get_images, eval_model, load_pretrained_model

    test_ds = load_data(
        conf["data_path"],
        mode="test",
        fourier=conf["fourier"],
        source_list=conf["source_list"],
    )

    num_images = conf["num_images"]
    rand = conf["random"]

    if num_images is None:
        num_images = len(test_ds)
    img_test, img_true = get_images(test_ds, num_images, norm_path="none", rand=rand)
    img_size = img_test.shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
    pred = eval_model(img_test, model, test=True)

    return pred, img_test, img_true

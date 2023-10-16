import numpy as np
import pytest
from scipy.stats import truncnorm


def truncnorm_moments(mu, sig, a, b):
    a, b = (a - mu) / sig, (b - mu) / sig
    sampled_gauss = truncnorm(a, b, loc=mu, scale=sig)

    return sampled_gauss.mean(), sampled_gauss.std()


@pytest.mark.order("last")
class TestEvaluation:
    def test_get_images(self):
        import torch

        from radionets.dl_framework.data import load_data
        from radionets.evaluation.utils import get_images

        test_ds = load_data(
            "./tests/build/data",
            mode="test",
            fourier=True,
        )

        num_images = 10
        rand = True
        img_size = test_ds[0][0][0].shape[-1]

        indices = torch.arange(num_images)
        assert len(indices) == 10

        if rand:
            indices = torch.randint(0, len(test_ds), size=(num_images,))
        img_test = test_ds[indices][0]

        assert img_test.shape == (10, 2, img_size, img_size)
        img_true = test_ds[indices][1]

        img_test, img_true, indices = get_images(test_ds, num_images, rand)

        assert img_true.shape == (10, 2, img_size, img_size)
        assert img_test.shape == (10, 2, img_size, img_size)
        assert len(indices) == 10

    def test_get_prediction(self):
        from pathlib import Path

        import toml

        from radionets.evaluation.train_inspection import get_prediction
        from radionets.evaluation.utils import read_config, save_pred

        config = toml.load("./tests/evaluate.toml")
        conf = read_config(config)

        img = get_prediction(conf)
        img_size = img["pred"].shape[-1]
        assert str(img["pred"].device) == "cpu"

        assert img["pred"].shape == (10, 2, img_size, img_size)
        assert img["inp"].shape == (10, 2, img_size, img_size)
        assert img["true"].shape == (10, 2, img_size, img_size)

        out_path = Path("./tests/build/test_training/evaluation/")
        out_path.mkdir(parents=True, exist_ok=True)
        save_pred(str(out_path) + "/predictions_model_eval.h5", img)

    def test_contour(self):
        import numpy as np
        import toml

        from radionets.evaluation.contour import area_of_contour
        from radionets.evaluation.utils import get_ifft, read_config, read_pred

        config = toml.load("./tests/evaluate.toml")
        conf = read_config(config)

        img = read_pred(
            "./tests/build/test_training/evaluation/predictions_model_eval.h5"
        )
        img_size = img["pred"].shape[-1]

        ifft_pred = get_ifft(img["pred"], amp_phase=conf["amp_phase"])
        ifft_truth = get_ifft(img["true"], amp_phase=conf["amp_phase"])

        assert ~np.isnan([ifft_pred, ifft_truth]).any()

        assert ifft_pred[0].shape == (img_size, img_size)
        assert ifft_truth[0].shape == (img_size, img_size)

        val = area_of_contour(ifft_pred[0], ifft_truth[0])

        assert isinstance(val, np.float64)
        assert ~np.isnan(val).any()
        assert val >= 0

    def test_im_to_array_value(self):
        from radionets.evaluation.jet_angle import im_to_array_value
        from radionets.evaluation.utils import read_pred

        img = read_pred(
            "./tests/build/test_training/evaluation/predictions_model_eval.h5"
        )

        image = img["pred"][0]
        img_size = image.shape[-1]

        x_coords, y_coords, value = im_to_array_value(image)

        assert x_coords.shape == (2, img_size**2)
        assert y_coords.shape == (2, img_size**2)
        assert value.shape == (2, img_size**2)

    def test_bmul(self):
        import torch

        from radionets.evaluation.jet_angle import bmul

        vec = torch.ones(1)
        mat = torch.ones(1, 2, 2)
        axis = 0

        cov = bmul(vec, mat, axis)

        assert cov.shape == (1, 2, 2)

    def test_pca(self):
        import toml
        import torch

        from radionets.evaluation.jet_angle import bmul, im_to_array_value, pca
        from radionets.evaluation.utils import get_ifft, read_config, read_pred

        config = toml.load("./tests/evaluate.toml")
        conf = read_config(config)

        torch.set_printoptions(precision=16)

        img = read_pred(
            "./tests/build/test_training/evaluation/predictions_model_eval.h5"
        )
        img_size = img["pred"].shape[-1]

        ifft_pred = get_ifft(img["pred"], conf["amp_phase"])
        assert ifft_pred.shape == (10, img_size, img_size)

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

        eig_vals_torch, eig_vecs_torch = torch.linalg.eigh(cov_w, UPLO="U")

        assert eig_vals_torch.shape == (10, 2)
        assert eig_vecs_torch.shape == (10, 2, 2)

        _, _, psi_torch = pca(torch.tensor(ifft_pred))

        assert len(psi_torch) == 10
        assert len(psi_torch[psi_torch > 360]) == 0

    def test_calc_jet_angle(self):
        import toml
        import torch

        from radionets.evaluation.jet_angle import calc_jet_angle
        from radionets.evaluation.utils import get_ifft, read_config, read_pred

        config = toml.load("./tests/evaluate.toml")
        conf = read_config(config)

        img = read_pred(
            "./tests/build/test_training/evaluation/predictions_model_eval.h5"
        )
        img_size = img["pred"].shape[-1]

        image = get_ifft(img["pred"], conf["amp_phase"])
        assert image.shape == (10, img_size, img_size)

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        image = image.clone()
        img_size = image.shape[-1]
        # ignore negative pixels, which can appear in predictions
        image[image < 0] = 0

        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        bs = image.shape[0]

        # only use brightest pixel
        max_val = torch.tensor([(i.max() * 0.4) for i in image])
        max_arr = (torch.ones(img_size, img_size, bs) * max_val).permute(2, 0, 1)
        image[image < max_arr] = 0

        assert image.shape == (10, img_size, img_size)

        m, n, alpha = calc_jet_angle(image)

        assert len(n) == 10
        assert len(alpha) == 10
        assert len(alpha[alpha > 360]) == 0

    def test_corners(self):
        import numpy as np

        from radionets.evaluation.blob_detection import corners

        r = np.random.randint(1, 10)
        x = np.random.randint(10, 40)
        y = np.random.randint(10, 40)

        x_coord, y_coord = corners(y, x, r)

        assert (x_coord[0] - x_coord[1]) % 2 == 1
        assert (y_coord[0] - y_coord[1]) % 2 == 1

    def test_calc_blobs_and_crop_first_comp(self):
        import numpy as np
        import toml
        import torch

        from radionets.evaluation.blob_detection import calc_blobs, crop_first_component
        from radionets.evaluation.utils import get_ifft, read_config, read_pred

        config = toml.load("./tests/evaluate.toml")
        conf = read_config(config)

        img = read_pred(
            "./tests/build/test_training/evaluation/predictions_model_eval.h5"
        )
        img_size = img["pred"].shape[-1]

        ifft_pred = get_ifft(torch.tensor(img["pred"][0]), conf["amp_phase"]).reshape(
            img_size, img_size
        )
        ifft_truth = get_ifft(torch.tensor(img["true"][0]), conf["amp_phase"]).reshape(
            img_size, img_size
        )

        blobs_pred, blobs_truth = calc_blobs(ifft_pred, ifft_truth)

        assert ~np.isnan(blobs_pred).any()
        assert ~np.isnan(blobs_truth).any()
        assert blobs_pred.all() >= 0
        assert blobs_truth.all() >= 0
        assert len(blobs_truth[0]) == 3

        flux_pred, flux_truth = crop_first_component(
            ifft_pred, ifft_truth, blobs_truth[0]
        )

        assert ~np.isnan(flux_pred).any()
        assert ~np.isnan(flux_truth).any()
        assert flux_pred.all() > 0
        assert flux_truth.all() > 0

    def test_gan_sources(self):
        import numpy as np
        import toml
        import torch

        from radionets.evaluation.train_inspection import evaluate_gan_sources
        from radionets.evaluation.utils import get_ifft, read_config, read_pred

        config = toml.load("./tests/evaluate.toml")
        conf = read_config(config)

        img = read_pred(
            "./tests/build/test_training/evaluation/predictions_model_eval.h5"
        )

        ifft_pred = get_ifft(torch.tensor(img["pred"][0]), conf["amp_phase"])
        ifft_truth = get_ifft(torch.tensor(img["true"][0]), conf["amp_phase"])

        img_size = ifft_pred.shape[-1]

        diff = (ifft_pred - ifft_truth).reshape(1, img_size, img_size)

        zero = np.isclose((np.zeros((1, img_size, img_size))), diff, atol=1e-3)
        assert zero.shape == (1, img_size, img_size)

        num_zero = zero.sum(axis=-1).sum(axis=-1) / (img_size * img_size) * 100
        assert num_zero >= 0
        assert num_zero <= 100
        assert ~np.isnan(num_zero)
        assert num_zero.dtype == "float64"

        ratio = diff.max(axis=-1).max(axis=-1) / ifft_truth.max(axis=-1).max(axis=-1)
        assert ratio.dtype == "float64"
        assert ratio > 0

        below_zero = np.sum(diff < 0, axis=(1, 2)) / (img_size * img_size) * 100
        above_zero = np.sum(diff > 0, axis=(1, 2)) / (img_size * img_size) * 100
        assert below_zero >= 0
        assert below_zero <= 100
        assert ~np.isnan(below_zero)
        assert below_zero.dtype == "float64"
        assert above_zero >= 0
        assert above_zero <= 100
        assert ~np.isnan(above_zero)
        assert above_zero.dtype == "float64"
        assert np.isclose(below_zero + above_zero, 100)

        assert evaluate_gan_sources(conf) is None

    def test_symmetry(self):
        import torch

        from radionets.dl_framework.model import symmetry

        x = torch.randint(0, 9, size=(1, 2, 4, 4))
        x_symm = symmetry(x.clone())
        for i in range(x.shape[-1]):
            for j in range(x.shape[-1]):
                assert (
                    x_symm[0, 0, i, j]
                    == x_symm[0, 0, x.shape[-1] - 1 - i, x.shape[-1] - 1 - j]
                )
                assert (
                    x_symm[0, 1, i, j]
                    == -x_symm[0, 1, x.shape[-1] - 1 - i, x.shape[-1] - 1 - j]
                )

        rot_amp = torch.rot90(x_symm[0, 0], 2)
        rot_phase = torch.rot90(x_symm[0, 1], 2)

        assert torch.isclose(rot_amp - x_symm[0, 0], torch.tensor(0)).all()
        assert torch.isclose(rot_phase + x_symm[0, 1], torch.tensor(0)).all()

    def test_sample_images(self):
        import numpy as np
        import torch
        import torch.nn.functional as F

        from radionets.evaluation.utils import get_ifft, read_pred, symmetry, trunc_rvs

        num_samples = 100
        num_img = 2
        img = read_pred("./tests/model/predictions_unc.h5")
        mean_amp, mean_phase = (
            img["pred"][:num_img, 0, :65, :],
            img["pred"][:num_img, 1, :65, :],
        )
        std_amp, std_phase = (
            img["unc"][:num_img, 0, :65, :],
            img["unc"][:num_img, 1, :65, :],
        )
        img_size = mean_amp.shape[-1]

        # amplitude
        sampled_gauss_amp = trunc_rvs(
            mean_amp,
            std_amp,
            mode="amp",
            num_samples=num_samples,
            target="cpu",
            nthreads=1,
        )

        # phase
        sampled_gauss_phase = trunc_rvs(
            mean_phase,
            std_phase,
            mode="phase",
            num_samples=num_samples,
            target="cpu",
            nthreads=1,
        )

        assert sampled_gauss_amp.shape == (
            num_img,
            num_samples,
            img_size // 2 + 1,
            img_size,
        )
        assert sampled_gauss_phase.shape == (
            num_img,
            num_samples,
            img_size // 2 + 1,
            img_size,
        )

        sampled_gauss_amp = sampled_gauss_amp.reshape(num_img * num_samples, 65, 128)
        sampled_gauss_phase = sampled_gauss_phase.reshape(
            num_img * num_samples, 65, 128
        )

        # masks
        mask_invalid_amp = sampled_gauss_amp < 0
        mask_invalid_phase = (sampled_gauss_phase <= (-np.pi - 1e-4)) | (
            sampled_gauss_phase >= (np.pi + 1e-4)
        )
        assert mask_invalid_amp.sum() == 0
        assert mask_invalid_phase.sum() == 0

        sampled_gauss = np.stack([sampled_gauss_amp, sampled_gauss_phase], axis=1)

        # pad resulting images and utilize symmetry
        sampled_gauss = F.pad(
            input=torch.tensor(sampled_gauss),
            pad=(0, 0, 0, 63),
            mode="constant",
            value=0,
        )
        sampled_gauss_symmetry = symmetry(sampled_gauss, None)

        fft_sampled_symmetry = get_ifft(
            sampled_gauss_symmetry, amp_phase=True, scale=False
        ).reshape(num_img, num_samples, 128, 128)

        results = {
            "mean": fft_sampled_symmetry.mean(axis=1),
            "std": fft_sampled_symmetry.std(axis=1),
        }
        assert results["mean"].shape == (num_img, img_size, img_size)
        assert results["std"].shape == (num_img, img_size, img_size)

    def test_uncertainty_plots(self):
        import toml

        from radionets.evaluation.utils import read_config, read_pred, sample_images

        config = toml.load("./tests/evaluate.toml")
        conf = read_config(config)
        img = read_pred("./tests/model/predictions_unc.h5")
        results = sample_images(
            img["pred"][:2, :, :65, :], img["unc"][:2, :, :65, :], 100, conf
        )
        assert results["mean"].shape == (2, 128, 128)
        assert results["std"].shape == (2, 128, 128)

    def test_analyse_intensity(self):
        import toml
        import torch

        from radionets.evaluation.contour import analyse_intensity
        from radionets.evaluation.utils import get_ifft, read_config, read_pred

        config = toml.load("./tests/evaluate.toml")
        conf = read_config(config)

        img = read_pred(
            "./tests/build/test_training/evaluation/predictions_model_eval.h5"
        )
        img_size = img["pred"].shape[-1]

        ifft_pred = get_ifft(torch.tensor(img["pred"][0]), conf["amp_phase"]).reshape(
            img_size, img_size
        )
        ifft_truth = get_ifft(torch.tensor(img["true"][0]), conf["amp_phase"]).reshape(
            img_size, img_size
        )

        sum_val, peak_val = analyse_intensity(ifft_pred, ifft_truth)

        assert len(sum_val) == 1
        assert len(peak_val) == 1
        assert sum_val > 0
        assert peak_val > 0

    def test_normalization(self):
        import numpy as np

        from radionets.evaluation.utils import (
            apply_normalization,
            read_pred,
            rescale_normalization,
        )

        img = read_pred(
            "./tests/build/test_training/evaluation/predictions_model_eval.h5"
        )
        norm_dict = {"all": 0}
        pred, norm_dict = apply_normalization(img["pred"], norm_dict)
        zeros = np.zeros(10)
        ones = np.ones(10)

        assert np.allclose(
            pred[:, 0].mean(axis=-1).mean(axis=-1), zeros, rtol=1e-4, atol=1e-5
        )
        assert np.allclose(
            pred[:, 1].mean(axis=-1).mean(axis=-1), zeros, rtol=1e-4, atol=1e-5
        )
        assert np.allclose(
            pred[:, 0].std(axis=-1).std(axis=-1), ones, rtol=1e-4, atol=1e-5
        )
        assert np.allclose(
            pred[:, 1].std(axis=-1).std(axis=-1), ones, rtol=1e-4, atol=1e-5
        )

        assert "means" in norm_dict
        assert "stds" in norm_dict

        pred = rescale_normalization(pred, norm_dict)

        assert np.allclose(
            pred[:, 0] - img["pred"][:, 0], np.zeros((128, 128)), rtol=1e-4, atol=1e-5
        )
        assert np.allclose(
            pred[:, 1] - img["pred"][:, 1], np.zeros((128, 128)), rtol=1e-4, atol=1e-5
        )

    def test_evaluation(self):
        import os
        import shutil

        from click.testing import CliRunner

        from radionets.evaluation.scripts.start_evaluation import main

        runner = CliRunner()
        result = runner.invoke(main, "tests/evaluate.toml")
        assert result.exit_code == 0

        if os.path.exists("tests/model/evaluation"):
            shutil.rmtree("tests/model/evaluation")


@pytest.mark.parametrize(
    "mode, target",
    [("phase", "cpu"), ("phase", "parallel"), ("amp", "cpu"), ("amp", "parallel")],
)
def test_trunc_rv(mode, target):
    from radionets.evaluation.utils import trunc_rvs

    mu = np.array([[0, 1], [1, 0]])
    sig = np.array([[0.5, 0.5], [1, 1]])
    nrand = int(1e5)

    if mode == "phase":
        a, b = -np.pi, np.pi
    elif mode == "amp":
        (a, b) = (0, np.inf)

    if target == "cpu":
        nthreads = 1
        with pytest.raises(ValueError):
            trunc_rvs(mu, sig, nrand, mode, target, nthreads=2)
    elif target == "parallel":
        nthreads = 2
        with pytest.raises(ValueError):
            trunc_rvs(mu, sig, nrand, mode, target, nthreads=1)

    with pytest.raises(ValueError):
        trunc_rvs(mu, sig, nrand, "phas", target, nthreads)

    with pytest.raises(ValueError):
        trunc_rvs(mu, sig, nrand, mode, "cp", nthreads)

    rvs = trunc_rvs(mu, sig, nrand, mode, target, nthreads)
    assert rvs.shape == (1, nrand, 2, 2)

    rvs = rvs.squeeze().reshape(nrand, 4)
    mu, sig = mu.flatten(), sig.flatten()

    for i in range(len(mu)):
        true_mu, true_sig = truncnorm_moments(mu[i], sig[i], a, b)
        assert np.isclose(np.mean(rvs[:, i]), true_mu, atol=1e-2, rtol=1e-2)
        assert np.isclose(np.std(rvs[:, i], ddof=1), true_sig, atol=1e-2, rtol=1e-2)
        assert np.max(rvs[:, i]) < b
        assert np.min(rvs[:, i]) > a

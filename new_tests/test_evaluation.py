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
        img_test, img_true = get_images(
            test_ds, num_images, norm_path="none", rand=rand
        )
        img_size = img_test.shape[-1]
        model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
        pred = eval_model(img_test, model, test=True)
        assert str(pred.device) == "cpu"

        # test for uncertainty
        if pred.shape[1] == 4:
            pred_1 = pred[:, 0, :].unsqueeze(1)
            pred_2 = pred[:, 2, :].unsqueeze(1)
            pred = torch.cat((pred_1, pred_2), dim=1)

        assert pred.shape == (10, 2, 63, 63)

    def test_contour(self):
        from radionets.dl_framework.data import load_data
        from radionets.dl_framework import architecture
        from radionets.evaluation.utils import (
            get_images,
            # load_pretrained_model,
            read_config,
            eval_model,
            get_ifft,
        )
        from radionets.evaluation.contour import (
            area_of_contour,
        )
        import toml
        import numpy as np
        import h5py

        import torch

        def load_pretrained_model(arch_name, model_path, img_size=63):
            if "filter_deep" in arch_name or "resnet" in arch_name:
                arch = getattr(architecture, arch_name)(img_size)
            else:
                arch = getattr(architecture, arch_name)()
            load_pre_model(arch, model_path, visualize=True)
            return arch

        def load_pre_model(learn, pre_path, visualize=True):
            checkpoint = torch.load(pre_path, map_location=torch.device("cpu"))
            if visualize:
                learn.load_state_dict(checkpoint["model"])

        def save_pred(path, x, y, name_x="x", name_y="y"):
            """
            write test data and predictions to h5 file
            x: truth of test data
            y: predictions of truth of test data
            """
            with h5py.File(path, "w") as hf:
                hf.create_dataset(name_x, data=x)
                hf.create_dataset(name_y, data=y)
                hf.close()

        def read_pred(path):
            """
            read data saved with save_pred from h5 file
            x: truth of test data
            y: predictions of truth of test data
            """
            with h5py.File(path, "r") as hf:
                x = np.array(hf["img_test"])
                y = np.array(hf["pred"])
                hf.close()
            return x, y

        config = toml.load("./new_tests/evaluate.toml")
        conf = read_config(config)

        test_ds = load_data(
            "./new_tests/build/data",
            mode="test",
            fourier=True,
            source_list=False,
        )

        num_images = 1
        rand = True

        img_test, img_true = get_images(
            test_ds, num_images, norm_path="none", rand=rand
        )
        img_size = img_test.shape[-1]
        model = load_pretrained_model(
            conf["arch_name"], "./new_tests/model/model_eval.model", img_size
        )
        pred = eval_model(img_test, model, test=True)
        assert str(pred.device) == "cpu"

        save_pred(
            "./new_tests/build/data/predictions.h5", img_test, pred, "img_test", "pred"
        )
        img_test1, pred1 = read_pred("./new_tests/build/data/predictions.h5")

        assert (np.array(img_test) == img_test1).all()
        assert (np.array(pred) == pred1).all()

        ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])
        ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])

        assert ~np.isnan([ifft_pred, ifft_truth]).any()

        val = area_of_contour(ifft_pred[0], ifft_truth[0])

        assert isinstance(val, np.float64)
        assert ~np.isnan(val).any()
        assert val > 0

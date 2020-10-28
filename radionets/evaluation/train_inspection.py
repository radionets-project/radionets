from pathlib import Path
from radionets.dl_framework.data import load_data
from radionets.evaluation.plotting import visualize_with_fourier, plot_results
from radionets.evaluation.utils import (
    reshape_2d,
    load_pretrained_model,
    get_images,
    eval_model,
)


def get_prediction(conf, num_images=None):
    test_ds = load_data(
        conf["data_path"],
        mode="test",
        fourier=conf["fourier"],
        source_list=conf["source_list"],
    )
    model = load_pretrained_model(conf["arch_name"], conf["model_path"])
    if num_images is None:
        num_images = len(test_ds)
    img_test, img_true = get_images(
        test_ds, num_images, norm_path=conf["norm_path"], rand=False
    )
    pred = eval_model(img_test, model)
    return pred, img_test, img_true


def create_inspection_plots(learn, train_conf, num_images=3, rand=False):
    pred, img_test, img_true = get_prediction(train_conf, num_images, rand)
    model_path = train_conf["model_path"]
    out_path = Path(model_path).parent
    if train_conf["fourier"]:
        for i in range(len(img_test)):
            visualize_with_fourier(
                i,
                img_test[i],
                pred[i],
                img_true[i],
                amp_phase=train_conf["amp_phase"],
                out_path=out_path,
            )
    else:
        plot_results(
            img_test.cpu(),
            reshape_2d(pred.cpu()),
            reshape_2d(img_true),
            out_path,
            save=True,
        )

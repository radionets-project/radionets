import numpy as np
from radionets.dl_framework.data import load_data
from radionets.evaluation.jets import fitgaussian_iterativ
from radionets.evaluation.plotting import (
    plot_jet_results,
    plot_jet_components_results,
    hist_jet_gaussian_distance,
    )
from radionets.evaluation.utils import load_pretrained_model
import toml
import torch
from tqdm import tqdm


torch.cuda.set_device(1)

config = toml.load("evaluate_jet.toml")
n = config["inspection"]["num_images"]
outpath = config["paths"]["outpath"]
h5_data = load_data(config["paths"]["data_path"], "test")
x = []
y = []
jet_list = []
print("Open the images")
for i in tqdm(range(100)):
    x.extend(h5_data.open_image("x", i))
    y.extend(h5_data.open_image("y", i))
    jet_list.extend(h5_data.open_image("list", i))
x = torch.stack(x)
y = torch.stack(y)
jet_list = torch.stack(jet_list)

model = load_pretrained_model(
    config["general"]["arch_name"], config["paths"]["model_path"], x.shape[-1]
)
model.eval()
pred = model(x).detach()
if not y.shape == pred.shape:
    raise ValueError(
        f"Shape of truth ({y.shape}) and prediction ({pred.shape}) is not equal. Program stopped."
    )

print("Plotting the results summed")
plot_jet_results(x[0:n], pred, y, path=outpath, save=True)
print("Plotting the results compenent-wise")
plot_jet_components_results(x[0:n], pred, y, path=outpath, save=True)
print("Plotting the iterative gaussian algorythm to predict components")
pred_for_gauss = torch.sum(pred[:, 0:-1], axis=1)
for i, pred_img in enumerate(pred_for_gauss[0:n]):
    fitgaussian_iterativ(pred_img, i, visualize=True, path=outpath, save=True)

all_params = []
for prediction in pred_for_gauss:
    all_params.append(fitgaussian_iterativ(prediction))

distances = []
for trues, preds in zip(jet_list, all_params):
    trues = trues[trues[:, 0] > 0.05]
    for pred in preds:
        diff = np.linalg.norm(trues[:, 1:3] - pred.parameters[1:3], axis=1)
        minimum = np.argmin(diff)
        distances.append((minimum, diff[minimum]))
distances = np.array(distances)

print("Plotting the histogram for iterative gaussian evaluation")
hist_jet_gaussian_distance(distances, path=outpath, save=True)
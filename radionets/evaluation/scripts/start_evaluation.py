import click
import numpy as np
import toml

from radionets.dl_framework.callbacks import PredictionImageGradient
from radionets.evaluation.train_inspection import (
    create_contour_plots,
    create_inspection_plots,
    create_predictions,
    create_source_plots,
    create_uncertainty_plots,
    evaluate_area,
    evaluate_area_sampled,
    evaluate_dynamic_range,
    evaluate_gan_sources,
    evaluate_intensity,
    evaluate_intensity_sampled,
    evaluate_mean_diff,
    evaluate_ms_ssim,
    evaluate_ms_ssim_sampled,
    evaluate_point,
    evaluate_unc,
    evaluate_viewing_angle,
    save_sampled,
)
from radionets.evaluation.utils import check_outpath, check_samp_file, read_config


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
def main(configuration_path):
    """
    Start evaluation of trained deep learning model.

    Parameters
    ----------
    configuration_path: str
        Path to the configuration toml file
    """
    conf = toml.load(configuration_path)
    eval_conf = read_config(conf)

    click.echo("\nEvaluation config:")
    print(eval_conf, "\n")

    if eval_conf["sample_unc"]:
        click.echo("Sampling test data set.\n")
        save_sampled(eval_conf)

    for entry in conf["inspection"]:
        if (
            conf["inspection"][entry] is not False
            and isinstance(conf["inspection"][entry], bool)
            and entry != "random"
        ):
            if (
                not check_outpath(eval_conf["model_path"])
                or conf["inspection"]["random"]
            ):
                create_predictions(eval_conf)
                break

    if eval_conf["unc"]:
        evaluate_unc(eval_conf)
        create_uncertainty_plots(
            eval_conf, num_images=eval_conf["num_images"], rand=eval_conf["random"]
        )
        click.echo(f"\nCreated {eval_conf['num_images']} uncertainty images.\n")

    if eval_conf["vis_pred"]:
        create_inspection_plots(
            eval_conf, num_images=eval_conf["num_images"], rand=eval_conf["random"]
        )

        click.echo(f"\nCreated {eval_conf['num_images']} test predictions.\n")

    if eval_conf["vis_ms_ssim"]:
        click.echo("\nVisualization of ms ssim is enabled for source plots.\n")

    if eval_conf["vis_dr"]:
        click.echo(f"\nCreated {eval_conf['num_images']} dynamic range plots.\n")

    if eval_conf["vis_source"]:
        create_source_plots(
            eval_conf, num_images=eval_conf["num_images"], rand=eval_conf["random"]
        )

        click.echo(f"\nCreated {eval_conf['num_images']} source predictions.\n")

    if eval_conf["plot_contour"]:
        create_contour_plots(
            eval_conf, num_images=eval_conf["num_images"], rand=eval_conf["random"]
        )

        click.echo(f"\nCreated {eval_conf['num_images']} contour plots.\n")

    if eval_conf["viewing_angle"]:
        click.echo("\nStart evaluation of viewing angles.\n")
        evaluate_viewing_angle(eval_conf)

    if eval_conf["dynamic_range"]:
        click.echo("\nStart evaluation of dynamic ranges.\n")
        evaluate_dynamic_range(eval_conf)

    if eval_conf["ms_ssim"]:
        click.echo("\nStart evaluation of ms ssim.\n")
        samp_file = check_samp_file(eval_conf)
        if samp_file:
            evaluate_ms_ssim_sampled(eval_conf)
        else:
            evaluate_ms_ssim(eval_conf)

    if eval_conf["intensity"]:
        click.echo("\nStart evaluation of intensity.\n")
        samp_file = check_samp_file(eval_conf)
        if samp_file:
            evaluate_intensity_sampled(eval_conf)
        else:
            evaluate_intensity(eval_conf)

    if eval_conf["mean_diff"]:
        click.echo("\nStart evaluation of mean difference.\n")
        evaluate_mean_diff(eval_conf)

    if eval_conf["area"]:
        click.echo("\nStart evaluation of the area.\n")
        samp_file = check_samp_file(eval_conf)
        if samp_file:
            evaluate_area_sampled(eval_conf)
        else:
            evaluate_area(eval_conf)

    if eval_conf["point"]:
        click.echo("\nStart evaluation of point sources.\n")
        evaluate_point(eval_conf)

    if eval_conf["predict_grad"]:
        output = PredictionImageGradient(
            test_data=eval_conf["data_path"],
            model=eval_conf["model_path"],
            amp_phase=eval_conf["amp_phase"],
            arch_name=eval_conf["arch_name"],
        )
        output = output.save_output_pred()
        grads_x, grads_y = output

        # specify names of saved gradients in x and y
        np.savetxt("grads_x.csv", grads_x, delimiter=",")
        np.savetxt("grads_y.csv", grads_y, delimiter=",")

        # # save image (no gradients)
        np.savetxt("test_img.csv", output, delimiter=",")

        # # save x and y grads for fourier amplitude and phase
        np.savetxt("grads_x_amp.csv", grads_x[0][0].cpu().numpy(), delimiter=",")
        np.savetxt("grads_x_phase.csv", grads_x[0][1].cpu().numpy(), delimiter=",")
        np.savetxt("grads_y_amp.csv", grads_y[0][0].cpu().numpy(), delimiter=",")
        np.savetxt("grads_y_phase.csv", grads_y[0][1].cpu().numpy(), delimiter=",")

    if eval_conf["gan"]:
        click.echo("\nStart evaluation of GAN sources.\n")
        evaluate_gan_sources(eval_conf)

import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_loss(learn, model_path, output_format="pdf"):
    """
    Plot train and valid loss of model.

    Parameters
    ----------
    learn: learner-object
        learner containing data and model
    model_path: str
        path to trained model
    """
    # to prevent the localhost error from happening first change the backende and
    # second turn off the interactive mode
    mpl.use("Agg")
    plt.ioff()
    # plt.style.use('./paper_small.rc')
    save_path = model_path.with_suffix("")
    print(f"\nPlotting Loss for: {model_path.stem}\n")
    logscale = learn.avg_loss.plot_loss()
    title = str(model_path.stem).replace("_", " ")
    plt.title(fr"{title}")
    if logscale:
        plt.yscale("log")
    plt.savefig(
        f"{save_path}_loss.{output_format}", bbox_inches="tight", pad_inches=0.01
    )
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_lr(learn, model_path, output_format="png"):
    """
    Plot learning rate of model.

    Parameters
    ----------
    learn: learner-object
        learner containing data and model
    model_path: str
        path to trained model
    """
    # to prevent the localhost error from happening first change the backende and
    # second turn off the interactive mode
    mpl.use("Agg")
    plt.ioff()
    save_path = model_path.with_suffix("")
    print(f"\nPlotting Learning rate for: {model_path.stem}\n")
    learn.avg_loss.plot_lrs()
    plt.savefig(f"{save_path}_lr.{output_format}", bbox_inches="tight", pad_inches=0.01)
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_lr_loss(learn, arch_name, out_path, skip_last, output_format="png"):
    """
    Plot loss of learning rate finder.

    Parameters
    ----------
    learn: learner-object
        learner containing data and model
    arch_path: str
        name of the architecture
    out_path: str
        path to save loss plot
    skip_last: int
        skip n last points
    """
    # to prevent the localhost error from happening first change the backende and
    # second turn off the interactive mode
    mpl.use("Agg")
    plt.ioff()
    print(f"\nPlotting Lr vs Loss for architecture: {arch_name}\n")
    learn.recorder.plot_lr_find()
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        out_path / f"lr_loss.{output_format}", bbox_inches="tight", pad_inches=0.01
    )
    mpl.rcParams.update(mpl.rcParamsDefault)

import subprocess
import tempfile


def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            # "--ExecutePreprocessor.timeout=1000",
            "--output",
            fout.name,
            path,
        ]
        subprocess.check_call(args)


def test():
    _exec_notebook("examples/01_dataset_simulation.ipynb")
    _exec_notebook("examples/02_model_training.ipynb")
    _exec_notebook("examples/03_model_evaluation.ipynb")
    _exec_notebook("examples/04_filter_deep.ipynb")

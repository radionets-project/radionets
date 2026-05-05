.. _eval_config:

**********************************************************
Evaluation Configuration (:mod:`radionets.io.eval_config`)
**********************************************************

The evaluation configuration system is built using `Pydantic <https://docs.pydantic.dev/>`_ which
automatically checks the types of the configuration parameters. Configurations can be
loaded from TOML files or instantiated directly in your code, which also allows using the configuration
system with external pipelines.

.. automodapi:: radionets.io.eval_config
   :no-inheritance-diagram:
   :skip: BaseModel


Loading the Configuration
-------------------------

Configurations are parsed and validated using the :class:`~radionets.io.eval_config.EvalConfig` class.
You can load a TOML file directly into this object:

.. code-block:: python

   from radionets.io.eval_config import EvalConfig


   # Load from TOML file
   config = EvalConfig.from_toml("path/to/eval_config.toml")

   # Accessing configuration values
   print(config.devices.accelerator)
   print(config.paths.data_path)


Alternatively, you can also instantiate the :class:`~radionets.io.eval_config.EvalConfig` class
directly and pass the options you want to change using a dictionary:

.. code-block:: python

   config = EvalConfig(**{"model": {"arch_name": ["SRResNet34"]}})

   # Will print the model architecture as
   # [<class 'radionets.architecture.archs.SRResNet34'>]
   print(config.model.arch_name)

The ``eval_config.toml`` file
-----------------------------

Below is a complete example of a typical ``eval_config.toml`` file with the default values.
You will only need to set the values you want to change. A comprehensive list of all available
options is shown further down below.

.. code-block:: toml

   title = "Evaluation configuration"

   [paths]
   data_path = "./example_data/"
   model_paths = ["./path/to/model.ckpt"]
   save_path = "./build"

   [model]
   arch_name = ["SRResNet18"]

   [devices]
   accelerator = ["auto"]
   num_devices = ["auto"]
   precision = ["32-true"]
   deepspeed = [false]
   strategy = ["auto"]

   [dataloader]
   module = "WebDatasetModule"
   batch_size = 100
   num_workers = 10
   prefetch_factor = 2
   persistent_workers = false
   fourier = true
   amp_phase = false

   [evaluation]
   viewing_angle = true
   dynamic_range = true
   intensity = true
   mean_diff = true
   area = true
   predict_grad = true
   evaluate_gan = true


Configuration Options Reference
-------------------------------
The configuration file is divided into several sections (TOML tables).

``[paths]``
^^^^^^^^^^^
Set the input and output directories for the evaluation process. Paths are expanded for the user,
so that, e.g., ``~`` is expanded to ``/home/<user>/``.

.. autopydantic_fields:: radionets.io.eval_config.PathsConfig

``[model]``
^^^^^^^^^^^
Here you can set the architecture and data representation.

.. autopydantic_fields:: radionets.io.eval_config.ModelConfig

``[devices]``
^^^^^^^^^^^^^
The devices table allows you to set the types of devices you want to use
and what strategy to use.

.. autopydantic_fields:: radionets.io.eval_config.DeviceConfig

``[dataloader]``
^^^^^^^^^^^^^^^^
The dataloader table contains all settings required to load the data,
and set the data representation or batch size.

.. autopydantic_fields:: radionets.io.eval_config.DataLoaderConfig

``[evaluation]``
^^^^^^^^^^^^^^^^
This table contains all available evaluation functions that ``radionets``
has to offer. Most of the metrics are ratios of predictions and
targets, such that

.. math::

   \text{Ratio} = \frac{\text{Prediction Metric}}{\text{Target Metric}}.


.. autopydantic_fields:: radionets.io.eval_config.EvaluationMethodsConfig

Radionets v0.4.1 (2025-08-08)
=============================


API Changes
-----------


Bug Fixes
---------


Data Model Changes
------------------


New Features
------------


Maintenance
-----------

- Updated ``CITATION.cff`` [`#190 <https://github.com/radionets-project/radionets/pull/190>`__]
- Updated authors [`#190 <https://github.com/radionets-project/radionets/pull/190>`__]


Refactoring and Optimization
----------------------------

Radionets v0.4.0 (2025-08-06)
=============================


API Changes
-----------


Bug Fixes
---------

- Replaced variance with standard deviation [`#155 <https://github.com/radionets-project/radionets/pull/155>`__]

- Computed uncertainty histogram on source pixels only

  - Check for existing sampling file before generating
  - Proper error propagation after using a normalization [`#164 <https://github.com/radionets-project/radionets/pull/164>`__]

- Fixed ``evaluate_msssim_sampled``

  - Fixed call of ``evaluate_msssim_sampled`` [`#171 <https://github.com/radionets-project/radionets/pull/171>`__]

- Fixed legend handles missing in eval contour plot [`#175 <https://github.com/radionets-project/radionets/pull/175>`__]


Data Model Changes
------------------


New Features
------------

- Added eval methods for sampled images [`#155 <https://github.com/radionets-project/radionets/pull/155>`__]

- Added zenodo.json [`#161 <https://github.com/radionets-project/radionets/pull/161>`__]

- Added new normalizing method which normalizes every image

  - Added saving options for more evaluation methods [`#165 <https://github.com/radionets-project/radionets/pull/165>`__]

- Added new block types for deep ResNets and UNets: :class:`~radionets.architecture.BottleneckResBlock`,
  :class:`~radionets.architecture.Encoder`, and :class:`~radionets.architecture.Decoder`

- Introduced new submodules :mod:`~radionets.architecture.activation`,
  :mod:`~radionets.architecture.archs`, and :mod:`~radionets.architecture.blocks`

    - Related classes (e.g., blocks, activation functions) and functions are moved
      from other submodules into these respective submodules
    - Improves readability and reusability

  - Added a logger that replaces bare prints throughout the code base
  - Added new diverging colormap ``radionets.PuOr`` that is used for plotting in radionets [`#178 <https://github.com/radionets-project/radionets/pull/178>`__]

- Added docs with API references and user/dev guides [`#187 <https://github.com/radionets-project/radionets/pull/187>`__]


Maintenance
-----------

- Use mamba in tests [`#159 <https://github.com/radionets-project/radionets/pull/159>`__]

- Added MANIFEST.in [`#160 <https://github.com/radionets-project/radionets/pull/160>`__]

- Cleaned up docs/changes [`#163 <https://github.com/radionets-project/radionets/pull/163>`__]

- Deleted unused functions

  - Deleted unused architectures
  - Renamed symmetry function [`#166 <https://github.com/radionets-project/radionets/pull/166>`__]

- Added radionets logo to README [`#169 <https://github.com/radionets-project/radionets/pull/169>`__]

- Fixed ``comet_ml`` callback

  - Update ``process_prediction`` with better if statement
  - Changed hardcoded values for sampling [`#170 <https://github.com/radionets-project/radionets/pull/170>`__]

- Set number of bins for histogram plotting [`#171 <https://github.com/radionets-project/radionets/pull/171>`__]

- Updated ``pyproject.toml`` and python version support

  - Switched to hatchling build backend [`#176 <https://github.com/radionets-project/radionets/pull/176>`__]

- Restructured ``dl_framework.architecture`` [`#177 <https://github.com/radionets-project/radionets/pull/177>`__]

- Flattened module hierarchy

  - Refactord architecture into modular components, see new features
  - Moved ``LocallyConnected2d`` class to :mod:`~radionets.architecture.unc_archs`
  - Refactored some callback submodule

    - Added error handling for cases where normalization attributes
      (``self.learn.normalize.mode``) may not be defined
    - Removed unnecessary calls to :func:`~radionets.simulations.visualize_simulations.create_OrBu`,
      replaced it with direct import of ``OrBu``

  - Refactored plotting tools and grouped plotting functions into logical groups
  - CI: Replaced ``pip`` with ``uv`` for package installation and added codecov test analytics [`#178 <https://github.com/radionets-project/radionets/pull/178>`__]


Refactoring and Optimization
----------------------------

- Added keyword for half of the image

  - Distinguish between tensor and array in get_ifft
  - Fixed micromamba installation [`#168 <https://github.com/radionets-project/radionets/pull/168>`__]

Radionets 0.3.0 (2023-08-04)
============================


API Changes
-----------


Bug Fixes
---------

- Fixed loading of correct sampling file [`#145 <https://github.com/radionets-project/radionets/pull/145>`__]

- Calculated normalization only on non-zero pixels

  - Fixed typo in rescaling operation [`#149 <https://github.com/radionets-project/radionets/pull/149>`__]

- Fixed sampling for images displayed in real and imaginary part [`#152 <https://github.com/radionets-project/radionets/pull/152>`__]


New Features
------------

- Enabled training and evaluation of half sized images (for 128 pixel images) [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]

- Added naming to save path, so that the files do not get overwritten as easily [`#144 <https://github.com/radionets-project/radionets/pull/144>`__]

- Added normalization callback with two different techniques

  - Updated plotting routines for real/imag images
  - Updated ``evaluate_area`` and ``evaluate_ms_ssim`` for half images
  - Added ``evaluate_ms_ssim`` for sampled images [`#146 <https://github.com/radionets-project/radionets/pull/146>`__]

- Add evaluation of intensity via peak flux and integrated flux comparison [`#150 <https://github.com/radionets-project/radionets/pull/150>`__]

- Centered bin on 1 for histogram evaluation plots

  - Added color to legend [`#151 <https://github.com/radionets-project/radionets/pull/151>`__]

- Added prettier labels and descriptions to plots [`#152 <https://github.com/radionets-project/radionets/pull/152>`__]


Maintenance
-----------

- Deleted unusable functions for new source types
- Deleted unused hardcoded scaling [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]

- Added masked loss functions
- Sorted bundles in simulations
- Minor adjustments in plotting scripts [`#141 <https://github.com/radionets-project/radionets/pull/141>`__]

- Consistent use of batch_size [`#142 <https://github.com/radionets-project/radionets/pull/142>`__]

- Added the model name to predictions and sampling file

  - Deleted unnecessary pad_unsqueeze function
  - Added amp_phase keyword to sample_images
  - Fixed deprecation warning in sampling.py
  - Added image size to test_evaluation.py routines [`#146 <https://github.com/radionets-project/radionets/pull/146>`__]

- Outsourced preprocessing steps in ``train_inspection.py`` [`#148 <https://github.com/radionets-project/radionets/pull/148>`__]

- Removed unused ``norm_path`` from all instances [`#153 <https://github.com/radionets-project/radionets/pull/153>`__]

- Deleted cropping

  - Updated colorbar label
  - Removed ``source_list`` argument [`#154 <https://github.com/radionets-project/radionets/pull/154>`__]


Refactoring and Optimization
----------------------------

- Optimized ``evaluation.utils.trunc_rvs`` with numba, providing functions compiled for cpu and parallel cpu computation. [`#143 <https://github.com/radionets-project/radionets/pull/143>`__]


Radionets 0.2.0 (2023-01-31)
============================


API Changes
-----------

- Train on half-sized images and applying symmetry afterward is a backward incompatible change
- Models trained with early versions of ``radionets`` are not supported anymore [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]


Bug Fixes
---------

- Fixed sampling of test data set
- Fixed same indices for plots [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]


New Features
------------

- Enabled training and evaluation of half sized images (for 128 pixel images) [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]


Maintenance
-----------

- Deleted unusable functions for new source types
- Deleted unused hardcoded scaling [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]


Refactoring and Optimization
----------------------------


Radionets 0.1.18 (2023-01-30)
=============================


API Changes
-----------


Bug Fixes
---------


New Features
------------

- Added creation of uncertainty plots
- Changed creation and saving/reading of predictions to ``dicts``

  - Prediction ``dicts`` have 3 or 4 entries depending on uncertainty

- Added scaled option to ``get_ifft``
- Created new dataset class for sampled images
- Created option for sampling and saving the whole test dataset
- Updated and wrote new tests [`#129 <https://github.com/radionets-project/radionets/pull/129>`__]


Maintenance
-----------

- Added and enabled ``towncrier`` in CI. [`#130 <https://github.com/radionets-project/radionets/pull/130>`__]

- Published ``radionets`` on pypi [`#134 <https://github.com/radionets-project/radionets/pull/134>`__]

- Updated README, used figures from the paper, minor text adjustments [`#136 <https://github.com/radionets-project/radionets/pull/136>`__]


Refactoring and Optimization
----------------------------

Radionets 0.3.0 (2023-08-04)
============================


API Changes
-----------


Bug Fixes
---------

- Fix loading of correct sampling file [`#145 <https://github.com/radionets-project/radionets/pull/145>`__]

- - calculate nomalization only on non-zero pixels
  - fix typo in rescaling operation [`#149 <https://github.com/radionets-project/radionets/pull/149>`__]

- fixed sampling for images displayed in real and imaginary part [`#152 <https://github.com/radionets-project/radionets/pull/152>`__]


New Features
------------

- enabled training and evaluation of half sized images (for 128 pixel images) [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]

- Add naming to save path, so that the files do not get overwritten as easily [`#144 <https://github.com/radionets-project/radionets/pull/144>`__]

- - Add normalization callback with two different techniques
  - Update plotting routines for real/imag images
  - Update evaluate_area and evaluate_ms_ssim for half images
  - Add evaluate_ms_ssim for sampled images [`#146 <https://github.com/radionets-project/radionets/pull/146>`__]

- add evaluation of intensity via peak flux and integrated flux comparison [`#150 <https://github.com/radionets-project/radionets/pull/150>`__]

- - centered bin on 1 for histogram evaluation plots
  - added color to legend [`#151 <https://github.com/radionets-project/radionets/pull/151>`__]

- add prettier labels and descriptions to plots [`#152 <https://github.com/radionets-project/radionets/pull/152>`__]


Maintenance
-----------

- Deleted unusable functions for new source types
  Deleted unused hardcoded scaling [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]

- add masked loss functions
  sort bundles in simulations
  minor adjustments in plotting scripts [`#141 <https://github.com/radionets-project/radionets/pull/141>`__]

- consistent use of batch_size [`#142 <https://github.com/radionets-project/radionets/pull/142>`__]

- - Add the model name to predictions and sampling file
  - Delete unnecessary pad_unsqueeze function
  - Add amp_phase keyword to sample_images
  - Fix deprecation warning in sampling.py
  - Add image size to test_evaluation.py routines [`#146 <https://github.com/radionets-project/radionets/pull/146>`__]

- Outsource preprocessing steps in `train_inspection.py` [`#148 <https://github.com/radionets-project/radionets/pull/148>`__]

- Remove unused `norm_path` from all instances [`#153 <https://github.com/radionets-project/radionets/pull/153>`__]

- - Deleted cropping
  - updated colorbar label
  - removed source_list argument [`#154 <https://github.com/radionets-project/radionets/pull/154>`__]


Refactoring and Optimization
----------------------------

- Optimize ``evaluation.utils.trunc_rvs`` with numba, providing functions compiled for cpu and parallel cpu computation. [`#143 <https://github.com/radionets-project/radionets/pull/143>`__]


Radionets 0.3.0 (2023-08-04)
============================


API Changes
-----------


Bug Fixes
---------

- Fix loading of correct sampling file [`#145 <https://github.com/radionets-project/radionets/pull/145>`__]

- - calculate nomalization only on non-zero pixels
  - fix typo in rescaling operation [`#149 <https://github.com/radionets-project/radionets/pull/149>`__]

- fixed sampling for images displayed in real and imaginary part [`#152 <https://github.com/radionets-project/radionets/pull/152>`__]


New Features
------------

- enabled training and evaluation of half sized images (for 128 pixel images) [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]

- Add naming to save path, so that the files do not get overwritten as easily [`#144 <https://github.com/radionets-project/radionets/pull/144>`__]

- - Add normalization callback with two different techniques
  - Update plotting routines for real/imag images
  - Update evaluate_area and evaluate_ms_ssim for half images
  - Add evaluate_ms_ssim for sampled images [`#146 <https://github.com/radionets-project/radionets/pull/146>`__]

- add evaluation of intensity via peak flux and integrated flux comparison [`#150 <https://github.com/radionets-project/radionets/pull/150>`__]

- - centered bin on 1 for histogram evaluation plots
  - added color to legend [`#151 <https://github.com/radionets-project/radionets/pull/151>`__]

- add prettier labels and descriptions to plots [`#152 <https://github.com/radionets-project/radionets/pull/152>`__]


Maintenance
-----------

- Deleted unusable functions for new source types
  Deleted unused hardcoded scaling [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]

- add masked loss functions
  sort bundles in simulations
  minor adjustments in plotting scripts [`#141 <https://github.com/radionets-project/radionets/pull/141>`__]

- consistent use of batch_size [`#142 <https://github.com/radionets-project/radionets/pull/142>`__]

- - Add the model name to predictions and sampling file
  - Delete unnecessary pad_unsqueeze function
  - Add amp_phase keyword to sample_images
  - Fix deprecation warning in sampling.py
  - Add image size to test_evaluation.py routines [`#146 <https://github.com/radionets-project/radionets/pull/146>`__]

- Outsource preprocessing steps in `train_inspection.py` [`#148 <https://github.com/radionets-project/radionets/pull/148>`__]

- Remove unused `norm_path` from all instances [`#153 <https://github.com/radionets-project/radionets/pull/153>`__]

- - Deleted cropping
  - updated colorbar label
  - removed source_list argument [`#154 <https://github.com/radionets-project/radionets/pull/154>`__]


Refactoring and Optimization
----------------------------

- Optimize ``evaluation.utils.trunc_rvs`` with numba, providing functions compiled for cpu and parallel cpu computation. [`#143 <https://github.com/radionets-project/radionets/pull/143>`__]


Radionets 0.2.0 (2023-01-31)
============================


API Changes
-----------

- train on half-sized iamges and applying symmetry afterward is a backward incompatible change
  models trained with early versions of `radionets` are not supported anymore [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]


Bug Fixes
---------

- fixed sampling of test data set
  fixed same indices for plots [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]


New Features
------------

- enabled training and evaluation of half sized images (for 128 pixel images) [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]


Maintenance
-----------

- Deleted unusable functions for new source types
  Deleted unused hardcoded scaling [`#140 <https://github.com/radionets-project/radionets/pull/140>`__]


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

- added creation of uncertainty plots
  changed creation and saving/reading of predictions to ``dicts``
  prediction ``dicts`` have 3 or 4 entries depending on uncertainty
  added scaled option to ``get_ifft``
  created new dataset class for sampled images
  created option for sampling and saving the whole test dataset
  updated and wrote new tests [`#129 <https://github.com/radionets-project/radionets/pull/129>`__]


Maintenance
-----------

- Add and enable ``towncrier`` in CI. [`#130 <https://github.com/radionets-project/radionets/pull/130>`__]

- publish radionets on pypi [`#134 <https://github.com/radionets-project/radionets/pull/134>`__]

- Update README, use figures from the paper, minor text adjustments [`#136 <https://github.com/radionets-project/radionets/pull/136>`__]


Refactoring and Optimization
----------------------------


Radionets 0.1.16 (2023-01-30)
=============================


API Changes
-----------


Bug Fixes
---------


New Features
------------

- added creation of uncertainty plots
  changed creation and saving/reading of predictions to ``dicts``
  prediction ``dicts`` have 3 or 4 entries depending on uncertainty
  added scaled option to ``get_ifft``
  created new dataset class for sampled images
  created option for sampling and saving the whole test dataset
  updated and wrote new tests [`#129 <https://github.com/radionets-project/radionets/pull/129>`__]


Maintenance
-----------

- Add and enable ``towncrier`` in CI. [`#130 <https://github.com/radionets-project/radionets/pull/130>`__]

- publish radionets on pypi [`#134 <https://github.com/radionets-project/radionets/pull/134>`__]


Refactoring and Optimization
----------------------------

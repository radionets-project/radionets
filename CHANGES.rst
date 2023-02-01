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

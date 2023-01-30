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

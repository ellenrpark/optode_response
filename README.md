# optode_response

This repository contains tools for predicting the response time for a suite of oxygen optodes:
- [Aanderaa 4330](https://www.aanderaa.com/media/pdfs/d378_aanderaa_oxygen_sensor_4330w_4330_4330f_low_en.pdf) (AA4330)
- [Aanderaa 4330 WTW](https://www.aanderaa.com/media/pdfs/d378_aanderaa_oxygen_sensor_4330w_4330_4330f_low_en.pdf) (AA4330|WTW)
- [RBRcoda T.ODO|slow](https://rbr-global.com/products/sensors/rbrcoda-todo/) (RBR|SLOW)
- [PyroScience PICO-O2-SUB](https://www.pyroscience.com/en/products/all-meters/pico-o2-sub) (PYRO-PICO)

Results are from the study: **Characterizing the response time of unpumped oxygen optodes for profiling applications**

Build python environment (pyoptode) from environment.yml file, as you need the visc function from the [gasex-python](https://github.com/boom-lab/gasex-python/tree/master) toolbox to predict response times.

Example code to predict and response time correct an optode time series is in responsetime_correction.ipynb. Response time results and prediction coefficients can be found in /data/.

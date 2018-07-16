# Autoencoding Variational Inference for Topic Models

---
## Code for the PAKDD 2018 paper: Autoencoding Variational Inference for Aspect Discovery.
---

---
#### Quick Start
---

This is a tensorflow implementation for Autoencoding Variational Inference for Aspect Discovery.

Download Dataset
`User Review Structure Analysis (URSA).zip` from https://drive.google.com/open?id=1Qnd7XRv_O6apd0ZRpeJJCf75NSIP7hCo

Extract to `data/`

---
To run the `prodLDA` model in the `URSA` dataset:

> `python run_aspects_analysis.py -m prodlda -f 100 -s 100 -t 3 -b 200 -r 0.0005 -e #NUMB_EPOCH`
---

---
##### OPTIONS
---

>`-m : prolda`
>`-b : batch size`
>`-r : learning rate`
>`-e : epoch number`


## Some hints to use when error
- https://github.com/llSourcell/Game-AI/issues/10 for bug "AttributionError: 'module' object has no attribute 'mul'" 
- type: echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc for bug "Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information" 

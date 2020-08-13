# KnowingOptimumValue_BO
Release code for ICML2020 Knowing The What But Not The Where in Bayesian Optimization

Copyright by the authors.


# Visualization
```
demo_visualization_knowing_the_what.....ipynb
```

# Running the algorithms in benchmark functions
```
demo_on_benchmark_functions.ipynb
```

# Customize your own black-box function
```
demo_customize_your_own_function.ipynb
```

# Running the comparison using the baselines in benchmark functions
```
run_all_benchmark_functions.py
```

After running these scripts to reproduce experiments, the results will be stored as pickles files in "pickle_storage" folder.
Then, we can plot all the results using scripts in the "plot" folder.

# Dependencies
* numpy
* scipy
* matplotlib


# Error with scipy=1.15
```
ValueError: `f0` passed has more than 1 dimension.
```
If this is the case, please downgrade to scipy=1.14.1

# Paper and Presentation
Visit https://proceedings.icml.cc/static/paper_files/icml/2020/2351-Paper.pdf


# Reference
```
Vu Nguyen and Michael A. Osborne.  "Knowing the what but not the where in Bayesian optimization." International Conference on Machine Learning (ICML), 2020.
```

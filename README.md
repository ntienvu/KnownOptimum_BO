<div align="center">
  
# Knowing The What But Not The Where in Bayesian Optimization


[![Paper](http://img.shields.io/badge/paper-arxiv.2006.07593-B31B1B.svg)](https://arxiv.org/pdf/1905.02685.pdf)
[![Conference](https://icml.cc/static/core/img/ICML-logo.svg)](https://icml.cc/Conferences/2021/ScheduleMultitrack?event=9843)
</div>

# Introduction
Bayesian optimization has demonstrated impressive success in finding the optimum input x∗ and output f∗ = f(x∗) = max f(x) of a black-box function f. In some applications, however, the optimum output f∗ is known in advance and the goal is to find the corresponding optimum input x∗. In this paper, we consider a new setting in BO in which the knowledge of the optimum output f∗ is available. Our goal is to exploit the knowledge about f∗ to search for the input x∗ efficiently. To achieve this goal, we first transform the Gaussian process surrogate using the information about the optimum output. Then, we propose two acquisition functions, called confidence bound minimization and expected regret minimization. We show that our approaches work intuitively and give quantitatively better performance against standard BO methods. We demonstrate real applications in tuning a deep reinforcement learning algorithm on the CartPole problem and XGBoost on Skin Segmentation dataset in which the optimum values are publicly available.


# Visualization
```
demo_visualization_knowing_the_what.....ipynb
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zo65EUzzmI883I5CNa2MWSoVhvEYRQWt#scrollTo=Ry9ylgpKVlgg)

# Running the algorithms in benchmark functions
```
demo_on_benchmark_functions.ipynb
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J2kI94DxDCBxwd85LHg6GxxJ5S91TmZn#scrollTo=rb3MrqPOQFyv)

# Customize your own black-box function
```
demo_customize_your_own_function.ipynb
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DbXQjeUZ6j05WoRU55te1cR_wOtq9U7Z#scrollTo=i2WzYqZfWD19)

# Running the comparison using the baselines in benchmark functions
```
run_all_benchmark_functions.py
```

After running these scripts to reproduce experiments, the results will be stored as pickles files in "pickle_storage" folder.
Then, we can plot all the results using scripts in the "plot" folder.

# Dependencies
* numpy >= 1.9.0
* scipy >= 0.14.0
* matplotlib >= 3.1.0
* scikit-learn >= 0.16.1

# Paper and Presentation
Visit https://proceedings.icml.cc/static/paper_files/icml/2020/2351-Paper.pdf


# Reference
```
Vu Nguyen and Michael A. Osborne.  "Knowing the what but not the where in Bayesian optimization." International Conference on Machine Learning (ICML), 2020.
```

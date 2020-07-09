from setuptools import setup, find_packages

setup(
    name='bayes_opt',
    version='1',
    packages = ["bayes_opt",
    "bayes_opt.test_functions",
	"bayes_opt.utility",
	"bayes_opt.visualization"],
    include_package_data = True,
    description='Bayesian Optimization Known Optimum Value',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.16.1",
    ],
)

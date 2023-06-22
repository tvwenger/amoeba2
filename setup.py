from setuptools import setup

setup(
    name="amoeba2",
    version="2.0",
    description="Automated Molecular Excitation Bayesian line-fitting Algorithm",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=["amoeba2"],
    install_requires=["numpy", "scipy", "matplotlib", "pymc", "corner", "scikit-learn"],
)

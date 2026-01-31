from setuptools import setup, find_packages

setup(
    name="datasets",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "torch==2.6.0",
        "torch-geometric==2.6.1",
        "torch-scatter==2.1.2+pt26cpu",
        "torch-sparse==0.6.18+pt26cpu",
        "torch-cluster==1.6.3+pt26cpu",
        "torch-spline-conv==1.2.2+pt26cpu",
        "numpy==2.3.1",
        "networkx==3.4.2",
        "pandas==2.3.1",
        "typing_extensions==4.12.2"
    ]

)

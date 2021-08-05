import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nima_keras",
    version="0.0.1",
    author="Somshubra Majumdar, Serhii Korzh",
    author_email="serhii.korzh@aalto.fi",
    description="Python Keras implementation of the different NIMA models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/souserge/neural-image-assessment",
    project_urls={
        "Bug Tracker": "https://github.com/souserge/neural-image-assessment/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=["tensorflow~=2.4.1", "keras", "numpy", "path", "h5py", "pillow"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)

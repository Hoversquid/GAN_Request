import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GAN_Request",
    version="0.0.1",
    author="Hoversquid",
    author_email="contactprojectworldmap@gmail.com",
    description="Used to take VQGAN requests to be queued on stream or for personal commissions.",
    long_description=long_description,
    url="https://github.com/Hoversquid/GAN_Request/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

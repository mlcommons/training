import setuptools

with open("README.md", "r") as f:
  long_description = f.read()

setuptools.setup(
    name="mlperf_compliance",
    version="0.0.1",
    author="Taylor Robie",
    author_email="taylorrobie@google.com",
    description="Tools for logging MLPerf compliance tags.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlperf/training/tree/master/compliance",
    packages=setuptools.find_packages(),
    classifiers=[
      "Programming Language :: Python",
      "LICENSE :: OSI APPROVED :: APACHE SOFTWARE LICENSE",
      "Operating System :: OS Independent",
    ],
)

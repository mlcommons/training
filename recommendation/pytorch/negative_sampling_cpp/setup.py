from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='negative_sampling',
      ext_modules=[CppExtension('negative_sampling', ['negative_sampling.cpp'])],
      cmdclass={'build_ext': BuildExtension})

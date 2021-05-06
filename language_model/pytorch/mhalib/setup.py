import torch
import setuptools
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mhalib',
    ext_modules=[
        CUDAExtension(
            name='mhalib',
            sources=['mha_funcs.cu'],
            extra_compile_args={
                               'cxx': ['-O3',],
                                'nvcc':['-O3','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', "--expt-relaxed-constexpr", "-ftemplate-depth=1024", '-gencode=arch=compute_70,code=sm_70','-gencode=arch=compute_80,code=sm_80','-gencode=arch=compute_80,code=compute_80']
                               }
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
})


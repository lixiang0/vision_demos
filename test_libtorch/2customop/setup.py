from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='customop',
    ext_modules=[
        CppExtension('customop', ['customop.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

# 尝试导入 pybind11，若未安装请先安装
try:
    import pybind11
except ImportError:
    print("pybind11 is not installed. Install with: pip install pybind11")
    sys.exit(1)

ext_modules = [
    Extension(
        'cpp_sampler',
        ['sampler.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name='cpp_sampler',
    version='0.1',
    author='Your Name',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
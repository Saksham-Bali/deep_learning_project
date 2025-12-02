from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy



import os
import platform

if os.name == 'nt':
    extra_compile_args = ["/openmp"]
    extra_link_args = []
else:
    extra_compile_args = ["-std=c++11", "-fopenmp"]
    extra_link_args = ["-std=c++11", "-fopenmp"]

ext_modules = [Extension(
       "nearest_neighbors",
       sources=["knn.pyx", "knn_.cxx",],  # source file(s)
       include_dirs=["./", numpy.get_include()],
       language="c++",            
       extra_compile_args = extra_compile_args,
       extra_link_args=extra_link_args,
  )]

setup(
    name = "KNN NanoFLANN",
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)

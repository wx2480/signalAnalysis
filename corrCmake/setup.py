from distutils.core import setup,Extension
from Cython.Build import cythonize
import numpy as np 

ext = Extension(name="paraCorr", sources=["paraCorr.pyx"])
setup(ext_modules=cythonize(ext,compiler_directives={'language_level' : "3"})
       ,include_dirs = [np.get_include()])
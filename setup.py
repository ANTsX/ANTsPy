import os
import re
import sys
import platform
import subprocess

import setuptools
from setuptools import find_packages
from setuptools import Extension#,setup, 
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from setuptools import distutils, Command, find_packages
import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.develop
import setuptools.command.build_py
import distutils.unixccompiler
import distutils.command.build
import distutils.command.clean

from distutils.core import setup

setup_py_dir = os.path.dirname(os.path.realpath(__file__))

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class install(setuptools.command.install.install):
    def run(self):
        setuptools.command.install.install.run(self)

class CMakeBuild(build_ext):
    def run(self):
        ## Find or Configure ITK ##
        print('Configuring ITK')
        if os.getenv('ITK_DIR'):
            print('Using Local ITK Installation at:\n %s' % os.getenv('ITK_DIR'))
        elif os.path.exists(os.path.join(setup_py_dir, 'itkbuild')):
            print('Using local ITK already built for this package')
            os.environ['ITK_DIR'] = os.path.join(setup_py_dir, 'itkbuild')
        else:
            print('No local ITK installation found... Building ITK now...')
            subprocess.check_call(['./configure_ITK.sh'], cwd=setup_py_dir)
            os.environ['ITK_DIR'] = os.path.join(setup_py_dir, 'itkbuild')

        ## Find or Configure ANTs ##
        print('Configuring ANTs core')
        subprocess.check_call(['./configure_ANTsPy.sh'], cwd=setup_py_dir)

        ## Configure ANTsPy library ##
        print('Configuring ANTsPy library')
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        setup_py_dir = os.path.dirname(os.path.realpath(__file__))
        extdir = os.path.join(setup_py_dir, 'ants/lib/')

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j3']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} {} -DVERSION_INFO=\\"{}\\"'.format("-Wno-inconsistent-missing-override",
                                                                    env.get('CXXFLAGS', ''),
                                                                    self.distribution.get_version())
        env['LINKFLAGS'] = '{}'.format("-Wno-inconsistent-missing-override")
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

long_description = open('README.md').read()

setup(
    name='antspy',
    version='0.1.3.dev8',
    author='Nicholas C. Cullen',
    author_email='nickmarch31@yahoo.com',
    description='Advanced Normalization Tools in Python',
    long_description=long_description,
    ext_modules=[CMakeExtension('ants', sourcedir=os.path.join(setup_py_dir,'ants/lib/'))],
    cmdclass={'build_ext':CMakeBuild, 'install':install},
    zip_safe=False,
    packages=find_packages(),
    package_data={'ants':['ants/lib/*.so*','ants/lib/*.so','lib/*.so*']},
    url='https://github.com/ANTsX/ANTsPy',
    classifiers=['Programming Language :: Python :: 3.6'],
)



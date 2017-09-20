import os
import re
import sys
import platform
import subprocess

from setuptools import find_packages
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
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
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      "-Wno-inconsistent-missing-override"]

        cfg = 'Debug' if self.debug else 'Release'
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
        env['LINKFlAGS'] = '{}'.format("-Wno-inconsistent-missing-override")
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup_py_dir = os.path.dirname(os.path.realpath(__file__))


## Find or Configure ITK ##
if os.getenv('ITK_DIR'):
    print('Using Local ITK Installation at:\n %s' % os.getenv('ITK_DIR'))
elif os.path.exists(os.path.join(setup_py_dir, 'itkbuild')):
    print('Using local ITK already built for this package')
    os.environ['ITK_DIR'] = os.path.join(setup_py_dir, 'itkbuild')
else:
    print('No local ITK installation found... Building ITK now...')
    subprocess.check_call(['./configure_ITK.sh'], cwd=setup_py_dir)
    os.environ['ITK_DIR'] = os.path.join(setup_py_dir, 'itkbuild')

## FIND or Configure VTK ##
#if os.getenv('VTK_DIR'):
#    print('Using Local VTK Installation at:\n %s' % os.getenv('VTK_DIR'))
#elif os.path.exists(os.path.join(setup_py_dir, 'vtkbuild')):
#    print('Using local VTK already built for this package')
#    os.environ['VTK_DIR'] = os.path.join(setup_py_dir, 'vtkbuild')
#else:
#    print('No local VTK installation found... Building VTK now...')
#    subprocess.check_call(['./configure_VTK.sh'], cwd=setup_py_dir)
#    os.environ['VTK_DIR'] = os.path.join(setup_py_dir, 'vtkbuild')


## Find or Configure ANTsPy library (ANTs Core, etc) ##
subprocess.check_call(['./configure_ANTsPy.sh'], cwd=setup_py_dir)

print('ITK_DIR: ' , os.getenv('ITK_DIR'))
#print('VTK_DIR: ' , os.getenv('VTK_DIR'))

setup(
    name='ants',
    version='0.1.2',
    author='Nicholas C. Cullen',
    author_email='nickmarch31@yahoo.com',
    description='Advanced Normalization Tools in Python',
    long_description='',
    ext_modules=[CMakeExtension('ants', sourcedir=os.path.join(setup_py_dir,'ants/lib/'))],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    packages=find_packages(),
    package_data={'ants':['ants/lib/*.so*','data/*','lib/*.so*']},
    classifiers=['Programming Language :: Python :: 3.6']
)

# add this second setup command as a temporary fix for when
# the compiled libraries do not get found initially
setup(
    name='ants',
    version='0.1.2',
    author='Nicholas C. Cullen',
    author_email='nickmarch31@yahoo.com',
    description='Advanced Normalization Tools in Python',
    long_description='',
    ext_modules=[CMakeExtension('ants', sourcedir=os.path.join(setup_py_dir,'ants/lib/'))],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    packages=find_packages(),
    package_data={'ants':['ants/lib/*.so*','data/*','lib/*.so*']},
    classifiers=['Programming Language :: Python :: 3.6']
)


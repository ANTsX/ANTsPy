import os
import platform
import re
import shutil
import subprocess
import sys
import time
from distutils.version import LooseVersion
from functools import cmp_to_key

import setuptools
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.install
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

setup_py_dir = os.path.dirname(os.path.realpath(__file__))
version = "0.3.5"  # ANTsPy version

if "--weekly" in sys.argv:
    sys.argv.remove("--weekly")
    version = "%sa0" % version


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class install(setuptools.command.install.install):
    def run(self):
        setuptools.command.install.install.run(self)


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command("build_ext")
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        global version, setup_py_dir
        print("-- Building version " + version)
        version_path = os.path.join(setup_py_dir, "ants", "version.py")
        with open(version_path, "w") as f:
            f.write("__version__ = '{}'\n".format(version))


class BuildExtFirst(setuptools.command.install.install):
    def run(self):
        self.run_command("build_py")
        return setuptools.command.install.install.run(self)


class CMakeBuild(build_ext):
    def run(self):
        ## Find or Configure VTK ##
        # print('Configuring VTK')
        # if os.getenv('VTK_DIR'):
        #    print('Using Local VTK Installation at:\n %s' % os.getenv('VTK_DIR'))
        # elif os.path.exists(os.path.join(setup_py_dir, 'vtkbuild')):
        #    print('Using local VTK already built for this package')
        #    os.environ['VTK_DIR'] = os.path.join(setup_py_dir, 'vtkbuild')
        # else:
        #    print('No local VTK installation found... Building VTK now...')
        #    subprocess.check_call(['./scripts/configure_VTK.sh'], cwd=setup_py_dir)
        #    os.environ['VTK_DIR'] = os.path.join(setup_py_dir, 'vtkbuild')

        ## Find or Configure ITK ##
        print("Configuring ITK")
        if os.getenv("ITK_DIR"):
            print("Using Local ITK Installation at:\n %s" % os.getenv("ITK_DIR"))
        elif os.path.exists(os.path.join(setup_py_dir, "itkbuild/ITKConfig.cmake")):
            print("Using local ITK already built for this package")
            os.environ["ITK_DIR"] = os.path.join(setup_py_dir, "itkbuild")
        else:
            print("No local ITK installation found... Building ITK now...")
            if platform.system() == "Windows":
                subprocess.check_call(
                    [".\\scripts\\configure_ITK.bat"], cwd=setup_py_dir
                )
            else:
                subprocess.check_call(["./scripts/configure_ITK.sh"], cwd=setup_py_dir)
            os.environ["ITK_DIR"] = os.path.join(setup_py_dir, "itkbuild")

        ## Find or Configure ANTs ##
        print("Configuring ANTs core")
        if platform.system() == "Windows":
            subprocess.check_call(
                [".\\scripts\\configure_ANTsPy.bat"], cwd=setup_py_dir
            )
        else:
            subprocess.check_call(["./scripts/configure_ANTsPy.sh"], cwd=setup_py_dir)

        ## Configure ANTsPy library ##
        print("Configuring ANTsPy library")
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = LooseVersion(
            re.search(r"version\s*([\d.]+)", out.decode()).group(1)
        )
        if cmake_version < "3.10.0":
            raise RuntimeError("CMake >= 3.10.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        setup_py_dir = os.path.dirname(os.path.realpath(__file__))
        extdir = os.path.join(setup_py_dir, "ants", "lib")

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        cfg = "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DBUILD_SHARED_LIBS:BOOL=OFF",
                "-DBUILD_ALL_ANTS_APPS:BOOL=OFF",
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
            ]
        #     cmake_args += [
        #         "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
        #     ]
        #     if sys.maxsize > 2 ** 32:
        #         cmake_args += ["-A", "x64"]
        #     build_args += ["--", "/m"]
        # else:
        #     cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        #     build_args += ["--", "-j2"]

        env = os.environ.copy()
        # if platform.system() != "Windows":
        #     env["CXXFLAGS"] = '{} {} -DVERSION_INFO=\\"{}\\"'.format(
        #         "-Wno-inconsistent-missing-override",
        #         env.get("CXXFLAGS", ""),
        #         self.distribution.get_version(),
        #     )
        #     env["LINKFLAGS"] = "{}".format("-Wno-inconsistent-missing-override")

        print(cmake_args)
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            [
                "cmake",
            ]
            + cmake_args
            + [
                ext.sourcedir,
            ],
            cwd=self.build_temp,
            env=env,
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


long_description = open("README.md").read()

setup(
    name="antspyx",
    version=version,
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "statsmodels",
        "webcolors",
        "matplotlib",
        "pyyaml",
        "chart_studio",
        "Pillow",
        "nibabel",
    ],
    author="Brian B. Avants and Nicholas Cullen",
    author_email="stnava@gmail.com",
    description="Advanced Normalization Tools in Python",
    long_description=long_description,
    long_description_content_type="text/markdown; charset=UTF-8; variant=GFM",
    ext_modules=[
        CMakeExtension("ants", sourcedir=os.path.join(setup_py_dir, "ants/lib/"))
    ],
    cmdclass={"build_py": build_py, "build_ext": CMakeBuild, "install": BuildExtFirst},
    zip_safe=False,
    packages=find_packages(),
    package_data={"ants": [
        "ants/lib/*.so*",
        "ants/lib/*.pyd",
        "ants/lib/*.dll",
        "lib/*.so*",
        "lib/*.pyd",
        "lib/*.dll",
        "ants/lib/*.so",
        "ants/lib/*.pyd",
        "ants/lib/*.dll",
        "lib/*.so"
        "lib/*.pyd"
        "lib/*.dll"
        ]},
    url="https://github.com/ANTsX/ANTsPy",
    classifiers=["Programming Language :: Python :: 3.6"],
)

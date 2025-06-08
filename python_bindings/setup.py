import os
import platform
from distutils.errors import CompileError, LinkError

import numpy as np
import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# 项目信息
__version__ = '0.1.0'
MODULE_NAME = 'deepsearch'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
BINDINGS_DIR = os.path.join(PROJECT_ROOT, "python_bindings")

# 平台信息
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


def find_cpp_sources(directory):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.endswith(".cpp")
    ]


def get_openmp_flags():
    if IS_WINDOWS:
        return ["/openmp"], []
    elif IS_MACOS:
        return ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
    elif IS_LINUX:
        return ["-fopenmp"], ["-fopenmp"]
    return [], []


import subprocess


def ensure_cmake_build():
    """确保CMake项目已构建，包括第三方库"""
    build_dir = os.path.join(PROJECT_ROOT, "build")

    try:
        # 检查build目录是否存在
        if not os.path.exists(build_dir):
            print("Creating build directory...")
            os.makedirs(build_dir)

        # 检查CMakeCache.txt是否存在，如果不存在则需要配置
        cmake_cache = os.path.join(build_dir, "CMakeCache.txt")
        if not os.path.exists(cmake_cache):
            print("Configuring CMake...")
            subprocess.run([
                "cmake",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_TESTS=OFF",
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                ".."
            ], cwd=build_dir, check=True)

        # 检查第三方库是否已编译
        spdlog_lib = os.path.join(build_dir, "_deps", "spdlog-build", "libspdlog.a")
        fmt_lib = os.path.join(build_dir, "_deps", "fmt-build", "libfmt.a")

        if not os.path.exists(spdlog_lib) or not os.path.exists(fmt_lib):
            print("Building third-party libraries...")
            # 先构建第三方库
            subprocess.run(["make", "spdlog", "fmt", "-j"], cwd=build_dir, check=True)

        print("CMake build setup completed.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during CMake build: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def get_cmake_target_info():
    """获取CMake构建的第三方库信息"""
    # 首先确保CMake项目已构建
    if not ensure_cmake_build():
        print("Warning: CMake build failed, falling back to system libraries")
        return [], [], [], []

    try:
        build_dir = os.path.join(PROJECT_ROOT, "build")
        deps_dir = os.path.join(build_dir, "_deps")

        include_dirs = []
        library_dirs = []
        libraries = []
        extra_objects = []  # 用于静态库

        # spdlog路径
        spdlog_src = os.path.join(deps_dir, "spdlog-src", "include")
        spdlog_build = os.path.join(deps_dir, "spdlog-build")

        if os.path.exists(spdlog_src):
            include_dirs.append(spdlog_src)

        # 查找spdlog静态库
        spdlog_lib = os.path.join(spdlog_build, "libspdlog.a")
        if os.path.exists(spdlog_lib):
            extra_objects.append(spdlog_lib)
            print(f"Found spdlog library: {spdlog_lib}")
        else:
            # 如果没有找到静态库，尝试动态库
            if os.path.exists(spdlog_build):
                library_dirs.append(spdlog_build)
                libraries.append("spdlog")

        # fmt路径
        fmt_src = os.path.join(deps_dir, "fmt-src", "include")
        fmt_build = os.path.join(deps_dir, "fmt-build")

        if os.path.exists(fmt_src):
            include_dirs.append(fmt_src)

        # 查找fmt静态库
        fmt_lib = os.path.join(fmt_build, "libfmt.a")
        if os.path.exists(fmt_lib):
            extra_objects.append(fmt_lib)
            print(f"Found fmt library: {fmt_lib}")
        else:
            if os.path.exists(fmt_build):
                library_dirs.append(fmt_build)
                libraries.append("fmt")

        return include_dirs, library_dirs, libraries, extra_objects

    except Exception as e:
        print(f"Warning: Could not get CMake target info: {e}")
        return [], [], [], []


class BuildExt(build_ext):
    user_options = build_ext.user_options + [('disable-openmp', None, "Disable OpenMP support")]

    def initialize_options(self):
        super().initialize_options()
        self.disable_openmp = False

    def finalize_options(self):
        super().finalize_options()
        self.openmp_include_dir = os.environ.get('OPENMP_INCLUDE_DIR')
        self.openmp_library_dir = os.environ.get('OPENMP_LIBRARY_DIR')
        if IS_MACOS:
            if not self.openmp_include_dir:
                self.openmp_include_dir = '/opt/homebrew/opt/libomp/include'
            if not self.openmp_library_dir:
                self.openmp_library_dir = '/opt/homebrew/opt/libomp/lib'

    def get_arch_flags(self):
        """获取适合当前平台的指令集优化标志"""
        # 用户自定义标志优先
        custom_flags = os.environ.get('ARCH_FLAGS', '')
        if custom_flags:
            return custom_flags.split()

        # 默认架构优化标志
        if IS_WINDOWS:
            return ['/arch:AVX2']  # MSVC 的 AVX2 选项

        flags = []
        machine = platform.machine().lower()

        # ARM 架构处理 (包括 Ubuntu ARM 和 macOS ARM)
        if 'arm' in machine or 'aarch' in machine:
            if IS_MACOS:
                return ['-mcpu=apple-m1', '-mtune=native']
            else:
                # Ubuntu ARM 的优化标志
                return [
                    '-march=armv8-a',  # ARMv8-A 基础指令集
                    '-mtune=native',  # 针对当前 CPU 优化
                ]

        # x86 架构处理
        flags.extend([
            '-msse3', '-msse4.1', '-msse4.2',
            '-mavx', '-mfma', '-mavx2',
            '-mbmi2', '-mpopcnt'
        ])

        return flags

    def build_extensions(self):
        cpp_flag = '/std:c++17' if IS_WINDOWS else '-std=c++17'

        # 获取指令集优化标志
        arch_flags = self.get_arch_flags()

        # 获取CMake构建的第三方库信息
        cmake_includes, cmake_lib_dirs, cmake_libs, extra_objects = get_cmake_target_info()

        for ext in self.extensions:
            ext.extra_compile_args = [cpp_flag]

            # 添加指令集优化标志
            ext.extra_compile_args.extend(arch_flags)

            ext.include_dirs.extend([
                pybind11.get_include(),
                np.get_include(),
                SRC_DIR
            ])

            # 添加CMake构建的第三方库
            ext.include_dirs.extend(cmake_includes)
            ext.library_dirs.extend(cmake_lib_dirs)
            ext.libraries.extend(cmake_libs)

            # 添加静态库对象文件
            if extra_objects:
                if not hasattr(ext, 'extra_objects'):
                    ext.extra_objects = []
                ext.extra_objects.extend(extra_objects)

            # 添加运行时库路径（仅对动态库）
            if cmake_lib_dirs:
                for lib_dir in cmake_lib_dirs:
                    ext.extra_link_args.append(f'-Wl,-rpath,{lib_dir}')
            if not IS_WINDOWS:
                ext.extra_compile_args += [
                    f'-DVERSION_INFO="{self.distribution.get_version()}"',
                    '-fvisibility=hidden'
                ]
            else:
                ext.extra_compile_args += [f'/DVERSION_INFO=\\"{self.distribution.get_version()}\\"']

            if not self.disable_openmp and self._check_openmp():
                compile_flags, link_flags = get_openmp_flags()
                ext.extra_compile_args += compile_flags
                ext.extra_link_args += link_flags

                # 包含和库路径设置
                if self.openmp_include_dir:
                    ext.include_dirs.append(self.openmp_include_dir)
                elif IS_MACOS:
                    ext.include_dirs.append('/opt/homebrew/opt/libomp/include')

                if self.openmp_library_dir:
                    ext.library_dirs.append(self.openmp_library_dir)
                elif IS_MACOS:
                    ext.library_dirs.append('/opt/homebrew/opt/libomp/lib')

        super().build_extensions()

    def _check_openmp(self):
        """尝试编译测试程序以检测 OpenMP 支持"""
        test_code = """
        #include <omp.h>
        int main() {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
            }
            return 0;
        }
        """
        try:
            tmp_dir = self.build_temp
            os.makedirs(tmp_dir, exist_ok=True)
            test_file = os.path.join(tmp_dir, "test_openmp.cpp")
            with open(test_file, "w") as f:
                f.write(test_code)

            compile_args, link_args = get_openmp_flags()
            if self.openmp_include_dir:
                compile_args += ["-I", self.openmp_include_dir]
            if self.openmp_library_dir:
                link_args += ["-L", self.openmp_library_dir]

            objs = self.compiler.compile([test_file], output_dir=tmp_dir, extra_postargs=compile_args)
            self.compiler.link_executable(objs, os.path.join(tmp_dir, "test_openmp_exec"), extra_postargs=link_args)
            return True
        except (CompileError, LinkError, Exception) as e:
            print(f"OpenMP support test failed: {e}")
            return False


# 构建 Extension 模块
source_files = [os.path.join(BINDINGS_DIR, "bindings.cpp")] + find_cpp_sources(SRC_DIR)

ext_modules = [
    Extension(
        MODULE_NAME,
        sources=source_files,
        include_dirs=[],  # 会在 build_ext 中添加
        language="c++"
    )
]

# 安装配置
setup(
    name=MODULE_NAME,
    version=__version__,
    description='Deep Approximate Nearest Neighbor Search',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='Koschei',
    author_email='nitianzero@gmail.com',
    url='https://github.com/kosthi/deepsearch',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    install_requires=[
        'numpy>=1.18',
        'pybind11>=2.6'
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache-2.0 License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    zip_safe=False,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'deepsearch-cli=deepsearch.cli:main',
        ],
    },
)

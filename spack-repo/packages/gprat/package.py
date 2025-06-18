# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


import sys

from spack.package import *

class Gprat(CMakePackage, CudaPackage):#, ROCmPackage):
    """Gaussian Process Regression using Asynchronous Task."""

    homepage = ""
    url = ""
    git = "https://github.com/SC-SGS/GPRat.git"
    maintainers("constracktor")

    license("MIT")

    version("main", branch="main")
    #version("0.3.0", sha256="")
    #version("0.2.0", sha256="")
    #version("0.1.0", sha256="")

    depends_on("cxx", type="build")

    map_cxxstd = lambda cxxstd: "2a" if cxxstd == "20" else cxxstd
    cxxstds = ("11", "14", "17", "20")
    variant(
        "cxxstd",
        default="17",
        values=cxxstds,
        description="Use the specified C++ standard when building.",
    )

    variant(
        "blas",
        default="openblas",
        description="Define CPU BLAS backend.",
        values=("openblas", "mkl"),
        multi=False,
    )


    variant("bindings", default=False, description="Build Python bindings")

    variant("examples", default=False, description="Build examples")

    variant("format", default=False, description="Build formating targets")


    # Build dependencies
    depends_on("git", type="build")
    depends_on("cmake@3.23:", type="build")
    depends_on("hpx@1.10.0: +static malloc=system networking=none max_cpu_count=256")

    # Backend dependecies
    depends_on("intel-oneapi-mkl shared=false", when="blas=mkl")
    depends_on("openblas fortran=false", when="blas=openblas")

    # CUDA
    depends_on("cuda +allow-unsupported-compilers", when="+cuda")
    depends_on("hpx@1.10.0: +cuda", when="+cuda")

    # ROCm not supported yet
    #depends_on("rocm", when="+rocm")

    # Only ROCm or CUDA maybe be enabled at once
    #conflicts("+rocm", when="+cuda")

    def cmake_args(self):
        spec, args = self.spec, []

        args += [
            self.define_from_variant("GPRAT_BUILD_BINDINGS", "bindings"),
            self.define_from_variant("GPRAT_ENABLE_EXAMPLES", "examples"),
            self.define_from_variant("GPRAT_ENABLE_FORMAT_TARGETS", "format"),
        ]
        if self.spec.satisfies("+cuda"):
            args += [self.define("GPRAT_WITH_CUDA", "ON")]
        else:
            args += [self.define("GPRAT_WITH_CUDA", "OFF")]

        if self.spec.satisfies("blas=mkl"):
            args += [self.define("GPRAT_ENABLE_MKL", "ON")]
        else:
            args += [self.define("GPRAT_ENABLE_MKL", "OFF")]

        return args

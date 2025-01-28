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
    git = "https://github.com/constracktor/GPRat.git"
    maintainers("constracktor")

    license("MIT")

    version("main", branch="main")

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
        "cpu_blas",
        default="mkl",
        description="Define CPU BLAS backend.",
    )


    variant("bindings", default=False, description="Build Python bindings")
    #variant("examples", default=False, description="Build examples")

    # Build dependencies
    depends_on("git", type="build")
    depends_on("cmake@3.23:", type="build")
    depends_on("hpx@1.9.0: +static malloc=system")

    # Other dependecies
    depends_on("intel-oneapi-mkl shared=false", when="cpu_blas=mkl")

    depends_on("cuda", when="+cuda")

    # ROCm not supported yet
    #depends_on("rocm", when="+rocm")

    # Only ROCm or CUDA maybe be enabled at once
    #conflicts("+rocm", when="+cuda")

    def cmake_args(self):
        spec, args = self.spec, []

        args += [
            self.define_from_variant("GPRAT_BUILD_BINDINGS", "bindings"),
            #self.define_from_variant("GPRAT_ENABLE_EXAMPLES", "examples"),
        ]

        return args

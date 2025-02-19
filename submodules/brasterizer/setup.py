from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="brasterizer",
    packages=['brasterizer'],
    ext_modules=[
        CUDAExtension(
            name="brasterizer._C",
            sources=[
            "csrc/rasterize.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                                        "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
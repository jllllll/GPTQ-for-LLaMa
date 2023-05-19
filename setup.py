from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="gptq-koboldai",
    version="0.0.4",
    install_requires=[
        "torch",
    ],
    extras_require = {
        "mpt_support": ["hf_bleeding_edge"]
    },
    packages=["gptq"],
    py_modules=["gptq"],
    ext_modules=[
        cpp_extension.CUDAExtension(
            "quant_cuda_v1", ["quant_cuda_v1/quant_cuda.cpp", "quant_cuda_v1/quant_cuda_kernel.cu"],
            extra_compile_args={'nvcc': ['-O3']},
        ),
        cpp_extension.CUDAExtension(
            "quant_cuda_v2", ["quant_cuda_v2/quant_cuda.cpp", "quant_cuda_v2/quant_cuda_kernel.cu"],
            extra_compile_args={'nvcc': ['-O3']},
        ),
        cpp_extension.CUDAExtension(
            "quant_cuda_v3", ["quant_cuda_v3/quant_cuda.cpp", "quant_cuda_v3/quant_cuda_kernel.cu"],
            extra_compile_args={'nvcc': ['-O3']},
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quantize_extension',
    ext_modules=[
        CUDAExtension(
            name='quantize_extension',
            sources=['bindings.cpp', 'quantize.cpp', 'quantize_kernel.cu', 'matmul.cpp', 'matmul_kernel.cu'],
            extra_compile_args={'cxx':['-O2'],
                                'nvcc':['-O2', '--use_fast_math', '-lcublas', '-gencode=arch=compute_86,code=sm_86']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


# rm -rf build/ dist/ *.egg-info/
# find . -type d -name "__pycache__" -exec rm -rf {} +
# python setup.py clean --all
# python setup.py build --force
# python setup.py install --force


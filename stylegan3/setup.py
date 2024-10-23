from setuptools import setup, find_packages

setup(
    name='stylegan3',
    version='0.1.0',
    author='NVIDIA',
    author_email='info@nvidia.com',
    description='StyleGAN3: Official PyTorch Implementation',
    packages=find_packages(),  # Automatically find and include packages
    install_requires=[
        # List the required packages here
        'torch>=1.7',  # Example, adjust as necessary
        'numpy',
        'scipy',
        'Pillow',
        # Add other dependencies based on requirements
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the required Python version
)

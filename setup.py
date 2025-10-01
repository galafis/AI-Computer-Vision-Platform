
from setuptools import setup, find_packages

setup(
    name='ai_computer_vision_platform',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.20.0',
        'opencv-python>=4.5.0',
        'scikit-learn>=0.24.0',
        'pandas>=1.2.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'pytest>=6.2.0',
        'Pillow>=8.0.0',
        'tqdm>=4.50.0',
        'scipy>=1.6.0',
        'imutils>=0.5.0',
        'dlib>=19.22.0',
        'face_recognition>=1.3.0',
        'tensorflow>=2.0.0', # Assuming a deep learning framework is used
        'keras>=2.0.0', # Assuming a deep learning framework is used
        'torch>=1.8.0', # Assuming a deep learning framework is used
        'torchvision>=0.9.0', # Assuming a deep learning framework is used
        'mediapipe>=0.8.0', # For pose and gesture estimation
    ],
    extras_require={
        'dev': [
            'flake8',
            'black',
            'isort',
            'mypy',
            'sphinx',
            'sphinx_rtd_theme',
            'twine',
        ],
    },
    author='AI Computer Vision Platform Team',
    author_email='contact@ai-cv-platform.com',
    description='Advanced Computer Vision Platform for data science and AI applications.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/GabrielLafis/AI-Computer-Vision-Platform',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.8',
)


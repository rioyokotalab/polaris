from setuptools import setup, find_packages


def get_long_description():
    with open('README.md') as f:
        readme = f.read()
    return readme


setup(
    name='polaris-py',
    packages=find_packages(),
    scripts=['bin/polaris-worker'],
    version='0.7.0',
    author='Shun Iwase, Linsho Kaku',
    author_email='s@sh8.io, kaku.l.aa@rio.gsic.titech.ac.jp',
    description='Polaris is a hyperparamter optimization library.',
    long_description=get_long_description(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6'
        ],
    platforms=['Linux', 'OS-X'],
    license='MIT',
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy',
        'pika==0.12.0',
        'statsmodels',
        'matplotlib',
    ],
    tests_require=['pytest'],
)

import sys

from setuptools import Extension, setup

if sys.version_info.major == 2:
    wrapper = 'src/wrapper27.c'
else:
    wrapper = 'src/wrapper35.c'


core = Extension(
    'puyocore',
    sources=[wrapper, 'src/core.c'],
    include_dirs=['src/include'],
)


if __name__ == '__main__':
    i_dont_know_how_tox_works = ['src/wrapper27.c', 'src/wrapper35.c', 'src/include/core.h']
    setup(
        setup_requires=['setuptools>=34.0', 'setuptools-gitver'],
        gitver=True,
        ext_modules=[core],
        scripts=i_dont_know_how_tox_works,
    )

from setuptools import setup

setup(
    name='focusaniso',
    version='0.1',
    py_modules=['focusaniso'],
    install_requires=['numpy'],
    extras_require={
        'test': ['pytest', 'pytest-cov'],
        'docs': ['sphinx'],
    },
)

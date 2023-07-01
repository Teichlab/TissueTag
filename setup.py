from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='tissue-tag',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'Pillow',
        'bokeh',
        'jupyter-bokeh',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-image',
        'tqdm',
    ],
    author='Oren Amsalem, Nadav Yayon',
    author_email='nadav.yayon@mail.huji.ac.il',
    description="Tissue Tag: jupyter image annotator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/nadavyayon/TissueTag',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)


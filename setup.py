from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='tissue-tag',
    version='0.2.2',
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
        'scikit-learn',
        'holoviews',
        'datashader',
        'panel==1.3.8',
        'jupyterlab'
    ],
    author='Oren Amsalem, Nadav Yayon, Andrian Yang',
    author_email='ny1@sanger.ac.uk',
    description="Tissue Tag: jupyter image annotator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/nadavyayon/TissueTag',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)


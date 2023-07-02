<p align="center">
	<img src="https://github.com/nadavyayon/TissueTag/blob/main/tissueTag_logo.png" width="300" >
</p>

# Tissue Tag jupyter image annotator
A jupyter-based image annotation tool - TissueTag is powered by the Bokeh python library (http://www.bokeh.pydata.org) and provides a simple annotation solution with subpixel resolution for fast interactive annotation of any image type or kind (brightfield, fluorescence, etc) as well as spatial omics. TissueTag generates discrete annotations (e.g. cortex, medulla etc) but can also output the euclidean distance of each spot/cell to the closest part of a given morphological structure, enabling continuous annotation. This thus holds spatial neighbourhood information that goes beyond the x-y coordinates of a given spot or cell. 

### We see this tool as a basic start and feel there are many useful applicaitons that could be added, so we welcome any contibution and look forward to suggestions!

## Installation

1) You need to install either [jupyter-lab](https://jupyter.org/install) or [jupyter-notebook](https://jupyter.org/install)
2) Install TissueTag using pip:
```
pip install tissue-tag
```
3) install kernel `ipython kernel install --name tissuetag --user`
## General instructions and examples:

1) When running on farm, you need to set the port (e.g., 5011) you are using to get it plot properly in the correct place.
2) To use anndata objects add scanpy to your env `pip install scanpy`


## How to cite:
preprint coming! stay tuned






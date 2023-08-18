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

### importing in a notebook 
`import tissue_tag as tt`

## How to cite:
preprint coming! stay tuned

## How to use 
We supply 2 examples of usage for TissueTag annotations: 
1) visium spatial transcriptomics
   in this example we annotate a postnatal thymus dataset by calling the major anatomimcal reagios based on either marker gene expression or manually, then training a random forst classifier for intial prediction follwed by manual corrections and migraiton of annotations back to the visium anndata object.
   [visium semi-automated tutorial](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/image_annotation_tutorial_visium_semi_automated.ipynb)
3) IBEX singel cell multiplex protein imaging
   in this example we annotate a postnatal thymus image by calling the major anatomimcal reagios and training a random forst classifier for intial prediction follwed by manual corrections
   [IBEX flourecent tutorial](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/image_annotation_tutorial_flourscent_final.ipynb)

## Usage on farm vs local machine 
Bokeh interactive plotting requiered communicaiton between the notebook instance and the browser. 
We have tested the functionality of TissueTag with jupyter lab or jupyter notebooks but have not yet implemented a solution for jupyter hub.
In addition SSH tunneling is not supported as well but VPN should work fine. 

When using a local machine the `show` function from bokeh should work with:
`show(app,notebook_url='localhost:8888')` 


When running on farm, you need to set the port (e.g., 5011) you are using to get it plot properly in the correct place.
when using a farm job with an interactive jupyter lab or jupyter notebook you should use this:
`show(app,notebook_url=f'{socket.gethostname()}:'+host)`
where `host` is your port number e.g. 5011

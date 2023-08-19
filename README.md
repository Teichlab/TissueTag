<p align="center">
	<img src="https://github.com/nadavyayon/TissueTag/blob/main/tissueTag_logo.png" width="300" >
</p>

# Tissue Tag jupyter image annotator
A jupyter-based image annotation tool - TissueTag is powered by the Bokeh python library (http://www.bokeh.pydata.org) and provides a simple annotation solution with subpixel resolution for fast interactive annotation of any image type or kind (brightfield, fluorescence, etc) as well as spatial omics. TissueTag generates discrete annotations (e.g. cortex, medulla etc) but can also output the euclidean distance of each spot/cell to the closest part of a given morphological structure, enabling continuous annotation. This thus holds spatial neighbourhood information that goes beyond the x-y coordinates of a given spot or cell.

## Tools 
1) Annotator: This tool allows for the interactive annotation of predefined anatomical objects through convex shape filling.
2) Scribbler: Aimed at "scribbling" broad labels on an image, this tool uses these labels to subsequently train a pixel classifier on the rest of the image.
3) Poly Annotator: Designed for labeling discrete, repetitive objects (e.g., lobules or separate compartments or cells), this tool generates polygons around objects and assigns labels to the discrete structures based on the number of objects (e.g., lobule_0, lobule_1, etc.).

### We see this tool as a basic start and feel there are many useful applications that could be added, so we welcome any contribution and look forward to suggestions!

## Installation

1) You need to install either [jupyter-lab](https://jupyter.org/install) or [jupyter-notebook](https://jupyter.org/install)
2) Install TissueTag using pip:
```
pip install tissue-tag
```
## How to use 
We supply 3 examples of usage for TissueTag annotations: 
1) visium spatial transcriptomics (manual) -  
   in this simple example we annotate a postnatal thymus dataset by calling the major anatomical regions manually and migration of the annotations back to the visium anndata object.
   [visium manual tutorial](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/image_annotation_tutorial_visium_manual_v1.ipynb)
2) visium spatial transcriptomics (semi automated) - in this more advanced sample we annotate a postnatal thymus dataset by calling the major anatomical regions based on marker gene expression, then training a random forest classifier for initial prediction followed by manual corrections and migration of the annotations back to the visium anndata object.
   [visium semi-automated tutorial](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/image_annotation_tutorial_visium_semi_automated.ipynb)
3) IBEX single cell multiplex protein imaging - 
   in this example we annotate a postnatal thymus image by calling the major anatomical regions and training a random forest classifier for initial prediction followed by manual corrections
   [IBEX fluorescent tutorial](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/image_annotation_tutorial_flourscent_final.ipynb)

## Usage on a cluster vs local machine 
Bokeh interactive plotting required communication between the notebook instance and the browser. 
We have tested the functionality of TissueTag with jupyter lab or jupyter notebooks but have not yet implemented a solution for jupyter hub.
In addition SSH tunnelling is not supported as well but if you are accessing the notebook from outside your institute, VPN access should work fine. 

When using a local machine the `show` function from bokeh should work with:
`show(app,notebook_url='localhost:8888')` 

When running on a cluster, you need to set the port (e.g., 5011) you are using to get it plot properly in the correct place.
when using a cluster job with an interactive jupyter lab or jupyter notebook you should use this:
`show(app,notebook_url=f'{socket.gethostname()}:'+host)`
where `host` is your port number e.g. 5011

## How to cite:
preprint coming! stay tuned



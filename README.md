

<p align="center">
	<img src="https://github.com/nadavyayon/TissueTag/blob/main/tissueTag_logo.png" width="300">
</p>

# TissueTag: Jupyter Image Annotator

TissueTag consists of two major components:
1) **Jupyter-based image annotation tool**:Utilising the Bokeh Python library (http://www.bokeh.pydata.org) empowered by Datashader (https://datashader.org/index.html) and holoviews (https://holoviews.org/index.html) for pyramidal image rendering. This tool offers a streamlined annotation solution with subpixel resolution for quick interactive annotation of various image types (e.g., brightfield, fluorescence). TissueTag produces labelled images* (e.g., cortex, medulla) and logs all tissue labels, and annotation resolution and colours for reproducibility.

2) **Mapping annotations to data**: This components converts the pixel based annotations to an hexagonal grid to allow calculation of distances between morphological structures offering continuous annotation. This calculation is dependent of the grid sampling frequency and the number of grid spots that we define as a spatial neighborhood. This is foundational for calculating a morphological axis (OrganAxis, see tutorials).

*Note: A labeled image is an integer array where each pixel value (0,1,2,...) corresponds to an annotated structure.*

**Annotator**: Enables interactive annotation of predefined anatomical objects via convex shape filling while toggling between reference and annotation image.

We envision this tool as a starting point, so contributions and suggestions are highly appreciated!

## Installation

1) You need to install either [jupyter-lab](https://jupyter.org/install) or [jupyter-notebook](https://jupyter.org/install)
2) Install TissueTag using pip:
```
pip install tissue-tag
```

## How to use 
We supply two examples of usage for TissueTag annotations: 
1) visium spatial transcriptomics -  
   in this example we annotate a postnatal thymus dataset by calling the major anatomical regions in multiple ways (based on marker gene expression or sparse manual annotations) then training a random forest pixel classifier for initial prediction followed by manual corrections [visium annotation tutorial](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/demo_visium_tissue_annotation.ipynb) and migration of the annotations back to the visium anndata object [mapping annotations to visium](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/demo_visium_annotation_to_spots.ipynb).
   We also show how to calculate a morphological axis (OrganAxis) in 2 ways. 

2) IBEX single cell multiplex protein imaging - 
   in this example we annotate a postnatal thymus image by calling the major anatomical regions and training a random forest classifier for initial prediction followed by manual corrections
   [IBEX annotation tutorial](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/demo_flourscent_map_annotations_to_cells.ipynb). Next, we show how one can migrate these annotations to segmented cells and calulcate a morphological axis (OrganAxis) [IBEX mapping annotations tutorial](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/demo_flourscent_map_annotations_to_cells.ipynb).
   
## Usage on a cluster vs local machine 
Bokeh interactive plotting required communication between the notebook instance and the browser. 
We have tested the functionality of TissueTag with jupyter lab or jupyter notebooks but have not yet found a solution that works for jupyter hub.
In addition SSH tunnelling is not supported as well but if you are accessing the notebook from outside your institute, VPN access should work fine. 

## How to cite:
please cite the following preprint - https://www.biorxiv.org/content/10.1101/2023.10.25.562925v1



<p align="center">
	<img src="https://github.com/nadavyayon/TissueTag/blob/main/tissueTag_logo.png" width="300">
</p>

# TissueTag: Jupyter Image Annotator

TissueTag consists of two major components:
1) **Jupyter-based image annotation tool**: Utilizing the Bokeh Python library (http://www.bokeh.pydata.org) empowered by Datashader (https://datashader.org/index.html) and holoviews (https://holoviews.org/index.html) for pyramidal image rendering. This tool offers a streamlined annotation solution with subpixel resolution for quick interactive annotation of various image types (e.g., brightfield, fluorescence). TissueTag produces labeled images (e.g., cortex, medulla) and logs all tissue labels, and annotation resolution and  colors for reproducibility.
2) **Mapping annotations to data**: This component facilitates the migration of annotations to spots/cells based on overlap with annotated structures. It also logs the minimum Euclidean distance of each spot/cell to the discrete annotations, offering continuous annotation. This contains spatial neighborhood information, adding to the x-y coordinates of a given spot or cell, and is foundational for calculating a morphological axis (OrganAxis, see tutorials).

*Note: A labeled image is an integer array where each pixel value (0,1,2,...) corresponds to an annotated structure.*

**Annotator**: Enables interactive annotation of predefined anatomical objects via convex shape filling while toggeling between reference and annotation image.

We envision this tool as a foundational starting point as its simplicity and transparent nature allows for many potential enhancements, additions and spinoffs. So contributions and suggestions are highly appreciated!

## Installation

1) You need to install either [jupyter-lab](https://jupyter.org/install) or [jupyter-notebook](https://jupyter.org/install)
2) Install TissueTag using pip:
```
pip install tissue-tag
```
## How to use 
We supply some examples of usage for TissueTag annotations: 
1) visium spatial transcriptomics -  
   in this example we annotate a postnatal thymus dataset by calling the major anatomical regions in multiple ways (based on marker gene expression or sprse manual annotations) then training a random forest classifier for initial prediction followed by manual corrections [visium annotation tutorial](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/demo_visium_annotation_git.ipynb) and migration of the annotations back to the visium anndata object [mapping annotations to visium](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/demo_visium_map_annotation_to_spots_git.ipynb).
   We also show how to calulcate a morphological axis (OrganAxis) in 2 ways. 
2) IBEX single cell multiplex protein imaging - 
   in this example we annotate a postnatal thymus image by calling the major anatomical regions and training a random forest classifier for initial prediction followed by manual corrections
   [IBEX annotation tutorial](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/demo_flourscent_annotation_git.ipynb). Next, we show how one can migrate these annotations to segmented cells and calulcate a morphological axis (OrganAxis) [IBEX mapping annotations tutorial](https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/demo_flourscent_map_annotations_to_cells_git.ipynb).
   
## Usage on a cluster vs local machine 
Bokeh interactive plotting required communication between the notebook instance and the browser. 
We have tested the functionality of TissueTag with jupyter lab or jupyter notebooks but have not yet implemented a solution for jupyter hub.
In addition SSH tunnelling is not supported as well but if you are accessing the notebook from outside your institute, VPN access should work fine. 

## How to cite:
please cite the following preprint - https://www.biorxiv.org/content/10.1101/2023.10.25.562925v1



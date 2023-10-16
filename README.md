<p align="center">
	<img src="https://github.com/nadavyayon/TissueTag/blob/main/tissueTag_logo.png" width="300">
</p>

# TissueTag: Jupyter Image Annotator

TissueTag consists of two major components:
1) **Jupyter-based image annotation tool**: Utilizing the Bokeh Python library (http://www.bokeh.pydata.org), this tool offers a streamlined annotation solution with subpixel resolution for quick interactive annotation of various image types (e.g., brightfield, fluorescence). TissueTag produces labeled images (e.g., cortex, medulla) and logs all tissue labels and annotation resolutions for reproducibility.
2) **Mapping annotations to data**: This component facilitates the migration of annotations to spots/cells based on overlap with annotated structures. It also logs the minimum Euclidean distance of each spot/cell to the discrete annotations, offering continuous annotation. This contains spatial neighborhood information, enriching the x-y coordinates of a given spot or cell, and is foundational for calculating a morphological axis (see tutorials).

*Note: A labeled image is an integer array where each pixel value (0,1,2,...) corresponds to an annotated structure.*

## Tools 
1) **Annotator**: Enables interactive annotation of predefined anatomical objects via convex shape filling.
2) **Scribbler**: Designed for broadly labeling an image. It uses these labels to train a pixel classifier on the remainder of the image.
3) **Poly Annotator**: Suited for labeling discrete, repetitive objects (e.g., lobules or separate compartments), this tool creates polygons around objects and labels them based on the object count (e.g., lobule_0, lobule_1).

We envision this tool as a foundational starting point. We believe there are many potential enhancements and additions, so contributions and suggestions are highly appreciated!

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



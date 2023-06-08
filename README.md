<p align="center">
	<img src="https://github.com/nadavyayon/TissueTag/blob/main/tissueTag%20logo-01.png" width="300" >
</p>


# Tissue Tag jupyter image annotator
General instructions until replaced with something proper 

1) You basically only need the `tissue_tag.py` file, which you will use to import the relevant functions like
 `import tissue_tag as tt` 
You also need the tutorial notebook https://github.com/nadavyayon/TissueTag/blob/main/Tutorials/image_annotation_tutorial.ipynb
2) You need to setup an env that has bokeh (https://github.com/bokeh/bokeh) and jupyter-bokeh (https://pypi.org/project/jupyter-bokeh/) installed. This is the trickiest bit sometimes 
3) When running on farm, you need to set the port (e.g., 5011) you are using to get it plot properly in the correct place 
Python package to interactively annotate histological images within a Jupyter notebook


# installation (partial - only tissue-tag) 

# installation (complete - imagespot) 

Instructions for imagespot env 

1) Create a new conda environment:
`conda create -n imagespot python=3.9`
say 'y' when asked

2) activate new environment 
`conda activate imagespot` 

3) install scanpy
`conda install -c conda-forge scanpy python-igraph leidenalg`

4) install jupyter-lab 
`conda install -c conda-forge jupyterlab`

6) install open-cv
`pip install opencv-python`

6_5) optional - install cellpose 
		`pip install cellpose` 
		`pip uninstall torch`
		`conda install pytorch cudatoolkit=11.3 -c pytorch`

7) install bokeh 
`pip install jupyter_bokeh`
`pip install bokeh` 

8)
`pip install scikit-image`


9) add the new enviroment to jupyter lab path  
$ ipython kernel install --name imagespot --user









import base64
import bokeh
import json
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
import scipy
import seaborn as sns
import skimage
import warnings
import os
from bokeh.models import FreehandDrawTool, PolyDrawTool, PolyEditTool,TabPanel, Tabs
from bokeh.plotting import figure, show
from functools import partial
from io import BytesIO
from packaging import version
from PIL import Image, ImageColor, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from scipy import interpolate
from scipy.spatial import distance
from skimage import data, feature, future, segmentation
from skimage.draw import polygon, disk
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


try:
    import scanpy as scread_visium
except ImportError:
    print('scanpy is not available')

Image.MAX_IMAGE_PIXELS = None

font_path = fm.findfont('DejaVu Sans')

def to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="png")
    data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{data}'

def create_icon(name, color):
    font_size = 25
    img = Image.new('RGBA', (30, 30), (255, 0, 0, 0))
    ImageDraw.Draw(img).text((5, 2), name, fill=tuple((np.array(matplotlib.colors.to_rgb(color)) * 255).astype(int)),
                             font=ImageFont.truetype(font_path, font_size))
    if version.parse(bokeh.__version__) < version.parse("3.1.0"):
        img = to_base64(img)
    return img

def read_image(
    path,
    ppm_image=None,
    ppm_out=1,
    contrast_factor=1,
    background_image_path=None,
):
    """
    Reads an H&E or fluorescent image and returns the image with optional enhancements.

    Parameters
    ----------
    path : str
        Path to the image. The image must be in a format supported by Pillow. Refer to
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html for the list
        of supported formats.
    ppm_image : float, optional
        Pixels per microns of the input image. If not provided, this will be extracted from the image 
        metadata with info['resolution']. If the metadata is not present, an error will be thrown.
    ppm_out : float, optional
        Pixels per microns of the output image. Defaults to 1.
    contrast_factor : int, optional
        Factor to adjust contrast for output image, typically between 2-5. Defaults to 1.
    background_image_path : str, optional
        Path to a background image. If provided, this image and the input image are combined 
        to create a virtual H&E (vH&E). If not provided, vH&E will not be performed.

    Returns
    -------
    numpy.ndarray
        The processed image.
    float
        The pixels per microns of the input image.
    float
        The pixels per microns of the output image.

    Raises
    ------
    ValueError
        If 'ppm_image' is not provided and cannot be extracted from the image metadata.
    """
    
    im = Image.open(path)
    if not(ppm_image):
        try:
            ppm_image = im.info['resolution'][0]
            print('found ppm in image metadata!, its - '+str(ppm_image))
        except:
            print('could not find ppm in image metadata, please provide ppm value')
    width, height = im.size
    newsize = (int(width/ppm_image*ppm_out), int(height/ppm_image*ppm_out))
    # resize
    im = im.resize(newsize,Image.Resampling.LANCZOS)
    im = im.convert("RGBA")
    #increase contrast
    enhancer = ImageEnhance.Contrast(im)
    factor = contrast_factor 
    im = enhancer.enhance(factor*factor)
    
    if background_image_path:
        im2 = Image.open(background_image_path)
        # resize
        im2 = im2.resize(newsize,Image.Resampling.LANCZOS)
        im2 = im2.convert("RGBA")
        #increase contrast
        enhancer = ImageEnhance.Contrast(im2)
        factor = contrast_factor 
        im2 = enhancer.enhance(factor*factor)
        # virtual H&E 
        # im2 = im2.convert("RGBA")
        im = simonson_vHE(np.array(im).astype('uint8'),np.array(im2).astype('uint8'))
        
    return np.array(im),ppm_image,ppm_out

def read_visium(
    spaceranger_dir_path,
    use_resolution='hires',
    res_in_ppm = None,
    fullres_path = None,
    header = None,
    plot = True,
):
    """
    Reads 10X Visium image data from SpaceRanger (v1.3.0).

    Parameters
    ----------
    spaceranger_dir_path : str
        Path to the 10X SpaceRanger output folder.
    use_resolution : {'hires', 'lowres', 'fullres'}, optional
        Desired image resolution. 'fullres' refers to the original image that was sent to SpaceRanger 
        along with sequencing data. If 'fullres' is specified, `fullres_path` must also be provided. 
        Defaults to 'hires'.
    res_in_ppm : float, optional
        Used when working with full resolution images to resize the full image to a specified pixels per 
        microns. 
    fullres_path : str, optional
        Path to the full resolution image used for mapping. This must be specified if `use_resolution` is 
        set to 'fullres'.
    header : int, optional (defa
        newer SpaceRanger could need this to be set as 0. Default is None. 
    plot : Boolean 
        if to plot the visium object to scale

    Returns
    -------
    numpy.ndarray
        The processed image.
    float
        The pixels per microns of the image.
    pandas.DataFrame
        A DataFrame containing information on the tissue positions.

    Raises
    ------
    AssertionError
        If 'use_resolution' is set to 'fullres' but 'fullres_path' is not specified.
    """

    spotsize = 55 #um spot size of a visium spot

    scalef = json.load(open(spaceranger_dir_path+'spatial/scalefactors_json.json','r'))
    if use_resolution=='fullres':
        assert fullres_path is not None, 'if use_resolution=="fullres" fullres_path has to be specified'
        
    df = pd.read_csv(spaceranger_dir_path+'spatial/tissue_positions_list.csv',header=header)
    if header==0: 
        df = df.set_index(keys='barcode')
        df = df[df['in_tissue']>0] # in tissue
         # turn df to mu
        fullres_ppm = scalef['spot_diameter_fullres']/spotsize
        df['pxl_row_in_fullres'] = df['pxl_row_in_fullres']/fullres_ppm
        df['pxl_col_in_fullres'] = df['pxl_col_in_fullres']/fullres_ppm
    else:
        df = df.set_index(keys=0)
        df = df[df[1]>0] # in tissue
         # turn df to mu
        fullres_ppm = scalef['spot_diameter_fullres']/spotsize
        df['pxl_row_in_fullres'] = df[4]/fullres_ppm
        df['pxl_col_in_fullres'] = df[5]/fullres_ppm
    
    
    if use_resolution=='fullres':
        im = Image.open(fullres_path)
        ppm = fullres_ppm
    else:
        im = Image.open(spaceranger_dir_path+'spatial/tissue_'+use_resolution+'_image.png')
        ppm = scalef['spot_diameter_fullres']*scalef['tissue_'+use_resolution+'_scalef']/spotsize
        
    
    if res_in_ppm:
        width, height = im.size
        newsize = (int(width*res_in_ppm/ppm), int(height*res_in_ppm/ppm))
        im = im.resize(newsize,Image.Resampling.LANCZOS)
        ppm = res_in_ppm
    
    # translate from mu to pixel
    df['pxl_col'] = df['pxl_col_in_fullres']*ppm
    df['pxl_row'] = df['pxl_row_in_fullres']*ppm
    

    im = im.convert("RGBA")

    if plot:
        coordinates = np.vstack((df['pxl_col'],df['pxl_row']))
        plt.imshow(im,origin='lower')
        plt.plot(coordinates[0,:],coordinates[1,:],'.')
        plt.title( 'ppm - '+str(ppm))
    
    return np.array(im), ppm, df


def scribbler(imarray, anno_dict, plot_size=1024):
    """
    Creates interactive scribble line annotations with Bokeh.

    Parameters
    ----------
    imarray : np.array
        Image in numpy array format.
    anno_dict : dict
        Dictionary of structures to annotate and colors for the structures.
    plot_size : int, optional
        Used to adjust the plotting area size. Default is 1024.

    Returns
    -------
    bokeh.plotting.Figure
        A Bokeh figure with interactive annotations.
    dict
        Dictionary of renderers for each annotation.
    """

    imarray_c = imarray.astype('uint8').copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])

    # Create a new bokeh figure
    p =  figure(width=int(plot_size), height=int(plot_size), match_aspect=True)

    # Add image to the figure
    p.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])

    render_dict = {}
    draw_tool_dict = {}
    for key in anno_dict.keys():
        render_dict[key] = p.multi_line([], [], line_width=5, alpha=0.4, color=anno_dict[key])
        draw_tool = FreehandDrawTool(renderers=[render_dict[key]], num_objects=200, icon=create_icon(key[0],anno_dict[key]))
        draw_tool.description = key
        p.add_tools(draw_tool)
        draw_tool_dict[key] = draw_tool

    return p, render_dict



def annotator(imarray, annotation, anno_dict, plot_size=1024,invert_y=False):
    """
    Interactive annotation tool with line annotations using Bokeh tabs for toggling between morphology and annotation.
    The principle is that selecting closed/semi-closed shapes that will later be filled according to the proper annotation.

    Parameters
    ----------
    imarray: np.array
        Image in numpy array format.
    annotation: np.array
        Label image in numpy array format.
    anno_dict: dict
        Dictionary of structures to annotate and colors for the structures.
    plot_size: int, default=1024
        Figure size for plotting.
    invert_y :boolean
        invert plot along y axis

    Returns
    -------
    Bokeh Tabs object
        Interactive tabs for annotating image.
    dict
        Dictionary of Bokeh renderers for each annotation.
    """
    from bokeh.models import ColumnDataSource, TabPanel, Tabs, FreehandDrawTool
    from bokeh.plotting import figure
    from bokeh.core.properties import field

    # Tab1
    imarray_c = annotation.copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])
    p = figure(width=plot_size, height=plot_size, match_aspect=True)
    p.quad(top=e[1:], bottom=e[:-1], left=0, right=h) if invert_y else None
    p.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    tab1 = TabPanel(child=p, title="Annotation")

    # Tab2
    imarray_c = imarray.copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])
    p1 = figure(width=plot_size, height=plot_size, match_aspect=True, x_range=p.x_range, y_range=p.y_range)
    p1.quad(top=e[1:], bottom=e[:-1], left=0, right=h) if invert_y else None
    p1.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    tab2 = TabPanel(child=p1, title="Image")

    render_dict = {}
    draw_tool_dict = {}
    source_data_dict = {}

    for l in anno_dict.keys():
        draw_source = ColumnDataSource(data=dict(xs=[], ys=[]))

        # For Tab1
        render_dict[l] = p.multi_line(field("xs"), field("ys"), line_width=3, alpha=0.4, color=anno_dict[l], source=draw_source)
        draw_tool_dict[l] = FreehandDrawTool(renderers=[render_dict[l]], num_objects=100, icon=create_icon(l[0], anno_dict[l]))
        draw_tool_dict[l].description = l
        p.add_tools(draw_tool_dict[l])

        # For Tab2
        render = p1.multi_line(field("xs"), field("ys"), line_width=3, alpha=0.4, color=anno_dict[l], source=draw_source)
        draw_tool = FreehandDrawTool(renderers=[render], num_objects=100, icon=create_icon(l[0], anno_dict[l]))
        draw_tool.description = l
        p1.add_tools(draw_tool)

        source_data_dict[l] = draw_source

    tabs = Tabs(tabs=[tab1, tab2])
    return tabs, render_dict


def complete_pixel_gaps(x,y):
    """
    Function to complete pixel gaps in a given x, y coordinates
    
    Parameters:
    x : list
        list of x coordinates
    y : list
        list of y coordinates
    
    Returns:
    new_x, new_y : tuple
        tuple of completed x and y coordinates
    """
    
    new_x_1 = []
    new_x_2 = []
    # iterate over x coordinate values
    for idx, px in enumerate(x[:-1]):
        # interpolate between each pair of x points
        interpolation = interpolate.interp1d(x[idx:idx+2], y[idx:idx+2])
        interpolated_x_1 = np.linspace(x[idx], x[idx+1], num=np.abs(x[idx+1] - x[idx] + 1))
        interpolated_x_2 = interpolation(interpolated_x_1).astype(int)
        # add interpolated values to new x lists
        new_x_1 += list(interpolated_x_1)
        new_x_2 += list(interpolated_x_2)

    new_y_1 = []
    new_y_2 = []
    # iterate over y coordinate values
    for idx, py in enumerate(y[:-1]):
        # interpolate between each pair of y points
        interpolation = interpolate.interp1d(y[idx:idx+2], x[idx:idx+2])
        interpolated_y_1 = np.linspace(y[idx], y[idx+1], num=np.abs(y[idx+1] - y[idx] + 1))
        interpolated_y_2 = interpolation(interpolated_y_1).astype(int)
        # add interpolated values to new y lists
        new_y_1 += list(interpolated_y_1)
        new_y_2 += list(interpolated_y_2)
    
    # combine x and y lists
    new_x = new_x_1 + new_y_2
    new_y = new_x_2 + new_y_1

    return new_x, new_y



def scribble_to_labels(imarray, render_dict, line_width=10):
    """
    Extract scribbles to a label image.
    
    Parameters
    ----------
    imarray: np.array
        Image in numpy array format used to calculate the label image size.
    render_dict: dict
        Bokeh object carrying annotations.
    line_width: int
        Width of the line labels.

    Returns
    -------
    np.array
        Annotation image.
    """
    annotations = {}
    training_labels = np.zeros((imarray.shape[1], imarray.shape[0]), dtype=np.uint8)

    for idx, a in enumerate(render_dict.keys()):
        xs = []
        ys = []
        annotations[a] = []

        for o in range(len(render_dict[a].data_source.data['xs'])):
            xt, yt = complete_pixel_gaps(
                np.array(render_dict[a].data_source.data['xs'][o]).astype(int),
                np.array(render_dict[a].data_source.data['ys'][o]).astype(int)
            )
            xs.extend(xt)
            ys.extend(yt)
            annotations[a].append(np.vstack([
                np.array(render_dict[a].data_source.data['xs'][o]).astype(int),
                np.array(render_dict[a].data_source.data['ys'][o]).astype(int)
            ]))

        xs = np.array(xs)
        ys = np.array(ys)
        inshape = (xs > 0) & (xs < imarray.shape[1]) & (ys > 0) & (ys < imarray.shape[0])
        xs = xs[inshape]
        ys = ys[inshape]
        
        training_labels[np.floor(xs).astype(int), np.floor(ys).astype(int)] = idx + 1
  
    training_labels = training_labels.transpose()
    return skimage.segmentation.expand_labels(training_labels, distance=line_width / 2)


def rgb_from_labels(labelimage, colors):
    """
    Helper function to plot from label images.
    
    Parameters
    ----------
    labelimage: np.array
        Label image with pixel values corresponding to labels.
    colors: list
        Colors corresponding to pixel values for plotting.

    Returns
    -------
    np.array
        Annotation image.
    """
    labelimage_rgb = np.zeros((labelimage.shape[0], labelimage.shape[1], 4))
    
    for c in range(len(colors)):
        color = ImageColor.getcolor(colors[c], "RGB")
        labelimage_rgb[labelimage == c + 1, 0:3] = np.array(color)

    labelimage_rgb[:, :, 3] = 255
    return labelimage_rgb.astype('uint8')



def sk_rf_classifier(im, training_labels):
    """
    A simple random forest pixel classifier from sklearn.
    
    Parameters
    ----------
    im : array
        The actual image to predict the labels from, should be the same size as training_labels.
    training_labels : array
        Label image with pixel values corresponding to labels.

    Returns
    -------
    array
        Predicted label map.
    """

    sigma_min = 1
    sigma_max = 16
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=False, texture=~True,
                            sigma_min=sigma_min, sigma_max=sigma_max, channel_axis=-1)

    features = features_func(im)
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                 max_depth=10, max_samples=0.05)
    clf = future.fit_segmenter(training_labels, features, clf)
    return future.predict_segmenter(features, clf)

def overlay_labels(im1, im2, alpha=0.8, show=True):
    """
    Helper function to merge 2 images.
    
    Parameters
    ----------
    im1 : array
        1st image.
    im2 : array
        2nd image.
    alpha : float, optional
        Blending factor, by default 0.8.
    show : bool, optional
        If to show the merged plot or not, by default True.

    Returns
    -------
    array
        The merged image.
    """

    #generate overlay image
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.dpi"] = 100
    out_img = np.zeros(im1.shape,dtype=im1.dtype)
    out_img[:,:,:] = (alpha * im1[:,:,:]) + ((1-alpha) * im2[:,:,:])
    out_img[:,:,3] = 255
    if show:
        plt.imshow(out_img,origin='lower')
    return out_img
   
def update_annotator(imarray, result, anno_dict, render_dict, alpha):
    """
    Updates annotations and generates overlay (out_img) and the label image (corrected_labels).
        
    Parameters
    ----------
    imarray : numpy.ndarray
        Image in numpy array format.
    result : numpy.ndarray
        Label image in numpy array format.
    anno_dict : dict
        Dictionary of structures to annotate and colors for the structures.
    render_dict : dict
        Bokeh data container.
    alpha : float
        Blending factor.
        
    Returns
    -------
    tuple
        Returns the overlay image and corrected labels as a tuple.
    """
    
    corrected_labels = result.copy()
    for idx, a in enumerate(render_dict.keys()):
        if render_dict[a].data_source.data['xs']:
            print(a)
            for o in range(len(render_dict[a].data_source.data['xs'])):
                x = np.array(render_dict[a].data_source.data['xs'][o]).astype(int)
                y = np.array(render_dict[a].data_source.data['ys'][o]).astype(int)
                rr, cc = polygon(y, x)
                inshape = np.where(np.array(result.shape[0] > rr) & np.array(0 < rr) & np.array(result.shape[1] > cc) & np.array(0 < cc))[0]
                corrected_labels[rr[inshape], cc[inshape]] = idx + 1 
                
    rgb = rgb_from_labels(corrected_labels, list(anno_dict.values()))
    out_img = overlay_labels(imarray, rgb, alpha=alpha, show=False)
    return out_img, corrected_labels


def rescale_image(label_image, target_size):
    """
    Rescales label image to original image size.
        
    Parameters
    ----------
    label_image : numpy.ndarray
        Labeled image.
    target_size : tuple
        Final dimensions.
        
    Returns
    -------
    numpy.ndarray
        Rescaled image.
    """
    imP = Image.fromarray(label_image)
    newsize = (target_size[0], target_size[1])
    return np.array(imP.resize(newsize))


def save_annotation(folder, label_image, file_name, anno_names, anno_colors, ppm):
    """
    Saves the annotated image as .tif and in addition saves the translation 
    from annotations to labels in a pickle file.
        
    Parameters
    ----------
    folder : str
        Folder where to save the annotations.
    label_image : numpy.ndarray
        Labeled image.
    file_name : str
        Name for tif image and pickle.
    anno_names : list
        Names of annotated objects.
    anno_colors : list
        Colors of annotated objects.
    ppm : float
        Pixels per microns.
    """
    
    label_image = Image.fromarray(label_image)
    label_image.save(folder + file_name + '.tif')
    with open(folder + file_name + '.pickle', 'wb') as handle:
        pickle.dump(dict(zip(range(1, len(anno_names) + 1), anno_names)), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + file_name + '_colors.pickle', 'wb') as handle:
        pickle.dump(dict(zip(anno_names, anno_colors)), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder + file_name + '_ppm.pickle', 'wb') as handle:
        pickle.dump({'ppm': ppm}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_annotation(folder, file_name, load_colors=False):
    """
    Loads the annotated image from a .tif file and the translation from annotations 
    to labels from a pickle file.

    Parameters
    ----------
    folder : str
        Folder path for annotations.
    file_name : str
        Name for tif image and pickle without extensions.
    load_colors : bool, optional
        If True, get original colors used for annotations. Default is False.

    Returns
    -------
    tuple
        Returns annotation image, annotation order, pixels per microns, and annotation color.
        If `load_colors` is False, annotation color is not returned.
    """
    
    imP = Image.open(folder + file_name + '.tif')

    ppm = imP.info['resolution'][0]
    im = np.array(imP)

    print(f'loaded annotation image - {file_name} size - {str(im.shape)}')
    with open(folder + file_name + '.pickle', 'rb') as handle:
        anno_order = pickle.load(handle)
        print('loaded annotations')        
        print(anno_order)
    with open(folder + file_name + '_ppm.pickle', 'rb') as handle:
        ppm = pickle.load(handle)
        print('loaded ppm')        
        print(ppm)
        
    if load_colors:
        with open(folder + file_name + '_colors.pickle', 'rb') as handle:
            anno_color = pickle.load(handle)
            print('loaded color annotations')        
            print(anno_color)
        return im, anno_order, ppm['ppm'], anno_color
    
    else:
        return im, anno_order, ppm['ppm']



def simonson_vHE(dapi_image, eosin_image):
    """
    Create virtual H&E images using DAPI and eosin images.
    from the developer website:
    The method is applied to data on a multiplex/Keyence microscope to produce virtual H&E images 
    using fluorescence data. If you find it useful, consider citing the relevant article:
    Creating virtual H&E images using samples imaged on a commercial multiplex platform
    Paul D. Simonson, Xiaobing Ren, Jonathan R. Fromm 
    doi: https://doi.org/10.1101/2021.02.05.21249150

    Parameters
    ----------
    dapi_image : ndarray
        DAPI image data.
    eosin_image : ndarray
        Eosin image data.

    Returns
    -------
    ndarray
        Virtual H&E image.
    """
    
    def createVirtualHE(dapi_image, eosin_image, k1, k2, background, beta_DAPI, beta_eosin):
        new_image = np.empty([dapi_image.shape[0], dapi_image.shape[1], 4])
        new_image[:,:,0] = background[0] + (1 - background[0]) * np.exp(- k1 * beta_DAPI[0] * dapi_image - k2 * beta_eosin[0] * eosin_image)
        new_image[:,:,1] = background[1] + (1 - background[1]) * np.exp(- k1 * beta_DAPI[1] * dapi_image - k2 * beta_eosin[1] * eosin_image)
        new_image[:,:,2] = background[2] + (1 - background[2]) * np.exp(- k1 * beta_DAPI[2] * dapi_image - k2 * beta_eosin[2] * eosin_image)
        new_image[:,:,3] = 1
        new_image = new_image*255
        return new_image.astype('uint8')

    k1 = k2 = 0.001

    background = [0.25, 0.25, 0.25]

    beta_DAPI = [9.147, 6.9215, 1.0]

    beta_eosin = [0.1, 15.8, 0.3]

    dapi_image = dapi_image[:,:,0]+dapi_image[:,:,1]
    eosin_image = eosin_image[:,:,0]+eosin_image[:,:,1]

    print(dapi_image.shape)
    return createVirtualHE(dapi_image, eosin_image, k1, k2, background, beta_DAPI, beta_eosin)

def generate_hires_grid(
    im,
    spot_diameter,
    pixels_per_micron,
):
    """
        creates an hexagonal grid of a specified size and density 
        
        Parameters
        ----------     
        im            
            image to fit the gri on (mostly for dimentions)
        spot_diameter  
            in microns - determines the spot size and thus the density of the grid
        pixels_per_micron  
            image resolution

    """
    
    helper = spot_diameter*pixels_per_micron
    X1 = np.linspace(helper,im.shape[0]-helper,round(im.shape[0]/helper))
    Y1 = np.linspace(helper,im.shape[1]-2*helper,round(im.shape[1]/(2*helper)))
    X2 = X1 + spot_diameter*pixels_per_micron/2
    Y2 = Y1 + helper
    Gx1, Gy1 = np.meshgrid(X1,Y1)
    Gx2, Gy2 = np.meshgrid(X2,Y2)
    positions1 = np.vstack([Gy1.ravel(), Gx1.ravel()])
    positions2 = np.vstack([Gy2.ravel(), Gx2.ravel()])
    positions = np.hstack([positions1,positions2])
    
    return positions


def grid_anno(
    im,
    annotation_image_list,
    annotation_image_names,
    annotation_label_list,
    spot_diameter,
    ppm_in,
    ppm_out,
    
):
    """
        transfer annotations to spot grid 
        
        Parameters
        ---------- 
        im 
            Original image (for resizing annotations)
        annotation_image_list
            list of images with annotations in the form of integer corresponding to labels   
        annotation_image_names
            list of image names with annotations in the form of strings corresponding to images 
        annotation_label_list
            list of dictionaries to convert label data to morphology
        spot_diameter
            same diameter used for grid
        ppm_in 
            pixels per micron for input image
        ppm_out 
            used to scale xy grid positions to original image

    """
    print('generating grid with spot size - '+str(spot_diameter)+', with resolution of - '+str(ppm_in)+' ppm')
    positions = generate_hires_grid(im,spot_diameter,ppm_in)
    positions = positions.astype('float32')
    dim = [im.shape[0],im.shape[1]]
    # transform tissue annotation images to original size

    radius = spot_diameter/4 # measure the annotation from the center of the spot 
    df = pd.DataFrame(
        np.vstack((np.array(range(len(positions.T[:,0]))),positions.T[:,0],
                   positions.T[:,1])).T,
        columns=['index','x','y'])
    for idx0,anno in enumerate(annotation_image_list):
        anno_orig = skimage.transform.resize(anno,dim,preserve_range=True).astype('uint8') 
        anno_dict = {}
        number_dict = {}
        name = f'{anno=}'.split('=')[0]
        print(annotation_image_names[idx0])
        for idx1,pointcell in tqdm(enumerate(positions.T)):
            disk = skimage.draw.disk([int(pointcell[1]),int(pointcell[0])],radius,shape=anno_orig.shape)
            anno_dict[idx1] = annotation_label_list[idx0][int(np.median(anno_orig[disk]))]
            number_dict[idx1] = int(np.median(anno_orig[disk]))
        df[annotation_image_names[idx0]] = anno_dict.values()
        df[annotation_image_names[idx0]+'_number'] = number_dict.values()
    # scale to original image coordinates
    df['x'] = df['x']*ppm_out/ppm_in
    df['y'] = df['y']*ppm_out/ppm_in
    df['index'] = df['index'].astype(int)
    df.set_index('index', drop=True, inplace=True)
    return df


def dist2cluster_fast(df,ppm, annotation, KNN=5, logscale=False):
    from scipy.spatial import cKDTree

    print('calculating distance matrix with cKDTree')

    points = np.vstack([df['x'],df['y']]).T
    categories = np.unique(df[annotation])

    Dist2ClusterAll = {c: np.zeros(df.shape[0]) for c in categories}

    for idx, c in enumerate(categories):
        indextmp = df[annotation] == c
        if np.sum(indextmp) > KNN:
            print(c)
            cluster_points = points[indextmp]
            tree = cKDTree(cluster_points)
            # Get KNN nearest neighbors for each point
            distances, _ = tree.query(points, k=KNN)
            # Store the mean distance for each point to the current category
            if KNN == 1:
                Dist2ClusterAll[c] = distances/ppm # No need to take mean if only one neighbor
            else:
                Dist2ClusterAll[c] = np.mean(distances, axis=1)/ppm

    for c in categories:              
        if logscale:
            df["L2_dist_log10_"+annotation+'_'+c] = np.log10(Dist2ClusterAll[c])
        else:
            df["L2_dist_"+annotation+'_'+c] = Dist2ClusterAll[c]

    return Dist2ClusterAll

    
def anno_to_cells(df_cells, x_col, y_col, df_grid, annotation='annotations', plot=True):
    """
    Maps tissue annotations to segmented cells by nearest neighbors.
    
    Parameters
    ----------
    df_cells : pandas.DataFrame
        Dataframe with cell data.
    x_col : str
        Name of column with x coordinates in df_cells.
    y_col : str
        Name of column with y coordinates in df_cells.
    df_grid : pandas.DataFrame
        Dataframe with grid data.
    annotation : str, optional
        Name of the column with annotations in df_grid. Default is 'annotations'.
    plot : bool, optional
        If true, plots the coordinates of the grid space and the cell space to make sure 
        they are aligned. Default is True.

    Returns
    -------
    df_cells : pandas.DataFrame
        Updated dataframe with cell data.
    """
    
    print('make sure the coordinate systems are aligned e.g. axes are not flipped') 
    a = np.vstack([df_grid['x'], df_grid['y']])
    b = np.vstack([df_cells[x_col], df_cells[y_col]])
    
    if plot:
        plt.figure(dpi=100, figsize=[10, 10])
        plt.title('cell space')
        plt.plot(b[0], b[1], '.', markersize=1)
        plt.show()
        
        df_grid_temp = df_grid.iloc[np.where(df_grid[annotation] != 'unassigned')[0], :].copy()
        aa = np.vstack([df_grid_temp['x'], df_grid_temp['y']])
        plt.figure(dpi=100, figsize=[10, 10])
        plt.plot(aa[0], aa[1], '.', markersize=1)
        plt.title('annotation space')
        plt.show()
    
    annotations = df_grid.columns[~df_grid.columns.isin(['x', 'y'])]
    
    for k in annotations:
        print('migrating - ' + k + ' to segmentations')
        df_cells[k] = scipy.interpolate.griddata(points=a.T, values=df_grid[k], xi=b.T, method='nearest')
  
    return df_cells


def anno_to_visium_spots(df_spots, df_grid, ppm, plot=True,how='nearest',max_distance=10e10):
    """
    Maps tissue annotations to Visium spots according to the nearest neighbors.
    
    Parameters
    ----------
    df_spots : pandas.DataFrame
        Dataframe with Visium spot data.
    df_grid : pandas.DataFrame
        Dataframe with grid data.
    ppm : float 
        scale of annotation vs visium
    plot : bool, optional
        If true, plots the coordinates of the grid space and the spot space to make sure 
        they are aligned. Default is True.
    how : string, optinal
        This determines how the association between the 2 grids is made from the scipy.interpolate.griddata function. Default is 'nearest'
    max_distance : int
        maximal distance where points are not migrated 

    Returns
    -------
    df_spots : pandas.DataFrame
        Updated dataframe with Visium spot data.
    """
    import numpy as np
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    
    print('Make sure the coordinate systems are aligned, e.g., axes are not flipped.') 
    a = np.vstack([df_grid['x'], df_grid['y']])
    b = np.vstack([df_spots['pxl_col_in_fullres'], df_spots['pxl_row_in_fullres']])*ppm
    
    if plot:
        plt.figure(dpi=100, figsize=[10, 10])
        plt.title('Spot space')
        plt.plot(b[0], b[1], '.', markersize=1)
        plt.show()
        
        plt.figure(dpi=100, figsize=[10, 10])
        plt.plot(a[0], a[1], '.', markersize=1)
        plt.title('Morpho space')
        plt.show()
    
    annotations = df_grid.columns[~df_grid.columns.isin(['x', 'y'])]
    
    for k in annotations:
        print('Migrating - ' + k + ' to segmentations.')
              
        # Interpolation
        df_spots[k] = griddata(points=a.T, values=df_grid[k], xi=b.T, method=how)
        
        # Create KDTree
        tree = cKDTree(a.T)
        
        # Query tree for nearest distance
        distances, _ = tree.query(b.T, distance_upper_bound=max_distance)
        # Mask df_spots where the distance is too high
        df_spots[k][distances==np.inf] = None
        # df_spots[k] = scipy.interpolate.griddata(points=a.T, values=df_grid[k], xi=b.T, method=how)
  
    return df_spots


def plot_grid(df, annotation, spotsize=10, save=False, dpi=100, figsize=(5,5), savepath=None):
    """
    Plots a grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to be plotted.
    annotation : str
        Annotation to be used in the plot.
    spotsize : int, optional
        Size of the spots in the plot. Default is 10.
    save : bool, optional
        If true, saves the plot. Default is False.
    dpi : int, optional
        Dots per inch for the plot. Default is 100.
    figsize : tuple, optional
        Size of the figure. Default is (5,5).
    savepath : str, optional
        Path to save the plot. Default is None.

    Returns
    -------
    None
    """

    plt.figure(dpi=dpi, figsize=figsize)

    ct_order = list((df[annotation].value_counts() > 0).keys())
    ct_color_map = dict(zip(ct_order, np.array(sns.color_palette("colorblind", len(ct_order)))[range(len(ct_order))]))

    sns.scatterplot(x='x', y='y', hue=annotation, s=spotsize, data=df, palette=ct_color_map, hue_order=ct_order)

    plt.grid(False)
    plt.title(annotation)
    plt.axis('equal')

    if save:
        if savepath is None:
            raise ValueError('The savepath must be specified if save is True.')

        plt.savefig(savepath + '/' + annotation.replace(" ", "_") + '.pdf')

    plt.show()

    
def poly_annotator(imarray, annotation, anno_dict, fig_screen_size=1024):
    """
    Interactive annotation tool with line annotations using Bokeh tabs for toggling between morphology and annotation.
    The principle is that selecting closed/semiclosed shaped that will later be filled according to the proper annotation.

    Parameters
    ----------
    imarray : numpy.ndarray
        Image in numpy array format.
    annotation : numpy.ndarray
        Label image in numpy array format.
    anno_dict : dict
        Dictionary of structures to annotate and colors for the structures.
    fig_screen_size : int, optional
        Plotting area size. Default is 1024.

    Returns
    -------
    bokeh.models.widgets.tabs.Tabs
        Bokeh tabs containing the annotation and image panels.
    dict
        Dictionary containing the Bokeh renderers for the annotation lines.
    """

    # Tab1
    imarray_c = annotation[:, :].copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])
    p = figure(width=int(fig_screen_size), height=int(fig_screen_size), match_aspect=True)
    plotted_image = p.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    tab1 = TabPanel(child=p, title="Annotation")

    # Tab2
    imarray_c = imarray[:, :].copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])
    p1 = figure(width=int(fig_screen_size), height=int(fig_screen_size), match_aspect=True, x_range=p.x_range, y_range=p.y_range)
    plotted_image = p1.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    tab2 = TabPanel(child=p1, title="Image")

    anno_color_map = anno_dict

    # Brushes
    render_dict = {}
    plot_tool_dict = {}
    for l in list(anno_dict.keys()):
        render_dict[l] = p.multi_line([], [], line_width=3, alpha=0.6, color=anno_color_map[l])
        plot_tool_dict[l] = PolyDrawTool(renderers=[render_dict[l]], num_objects=300, icon=create_icon(l[0], anno_color_map[l]))
        plot_tool_dict[l].description = l
        p.add_tools(plot_tool_dict[l])

    tabs = Tabs(tabs=[tab1, tab2])
    return tabs, render_dict


def object_annotator(imarray, result, anno_dict, render_dict, alpha):
    """
    Extracts annotations and labels them according to brush strokes while generating out_img and the label image corrected_labels, and the anno_dict object.

    Parameters
    ----------
    imarray : numpy.ndarray
        Image in numpy array format.
    result : numpy.ndarray
        Label image in numpy array format.
    anno_dict : dict
        Dictionary of structures to annotate and colors for the structures.
    render_dict : dict
        Bokeh data container.
    alpha : float
        Blending factor.

    Returns
    -------
    numpy.ndarray
        Corrected label image.
    dict
        Dictionary containing the object colors.
    """

    colorpool = ['green', 'cyan', 'brown', 'magenta', 'blue', 'red', 'orange']

    result[:] = 1
    corrected_labels = result.copy()
    object_dict = {'unassigned': 'yellow'}

    for idx,a in enumerate(render_dict.keys()):
        if render_dict[a].data_source.data['xs']:
            print(a)
            for o in range(len(render_dict[a].data_source.data['xs'])):
                x = np.array(render_dict[a].data_source.data['xs'][o]).astype(int)
                y = np.array(render_dict[a].data_source.data['ys'][o]).astype(int)
                rr, cc = polygon(y, x)
                inshape = (result.shape[0]>rr) & (0<rr) & (result.shape[1]>cc) & (0<cc)         # make sure pixels outside the image are ignored
                corrected_labels[rr[inshape], cc[inshape]] = o+2
                object_dict[a+'_'+str(o)] = random.choice(colorpool)

    return corrected_labels, object_dict


def gene_labels(adata, df, training_labels, marker_dict, annodict, r, labels_per_marker):
    """
    Assign labels to training spots based on gene expression.

    Parameters
    ----------
    adata : scanpy.AnnData
        Annotated data matrix.
    df : pandas.DataFrame
        DataFrame containing spot coordinates.
    training_labels : numpy.ndarray
        Array for storing the training labels.
    marker_dict : dict
        Dictionary mapping markers to genes.
    annodict : dict
        Dictionary mapping markers to annotation names.
    r : float
        Radius of the spots.
    labels_per_marker : dict
        Dictionary mapping markers to the number of labels per marker.

    Returns
    -------
    numpy.ndarray
        Array containing the training labels.
    """

    import scanpy

    for m in list(marker_dict.keys()):
        print(marker_dict[m])
        GeneIndex = np.where(adata.var_names.str.fullmatch(marker_dict[m]))[0]
        scanpy.pp.normalize_total(adata)
        GeneData = adata.X[:, GeneIndex].todense()
        SortedExp = np.argsort(GeneData, axis=0)[::-1]
        list_gene = adata.obs.index[np.array(np.squeeze(SortedExp[range(labels_per_marker[m])]))[0]]
        for idx, sub in enumerate(list(annodict.keys())):
            if sub == m:
                back = idx
        for coor in df.loc[list_gene, ['pxl_row','pxl_col']].to_numpy():
            training_labels[disk((coor[0], coor[1]), r)] = back + 1
    return training_labels


def background_labels(shape, coordinates, r, every_x_spots=10, label=1):
    """
    Generate background labels.

    Parameters
    ----------
    shape : tuple
        Shape of the training labels array.
    coordinates : numpy.ndarray
        Array containing the coordinates of the spots.
    r : float
        Radius of the spots.
    every_x_spots : int, optional
        Spacing between background spots. Default is 10.
    label : int, optional
        Label value for background spots. Default is 1.

    Returns
    -------
    numpy.ndarray
        Array containing the background labels.
    """

    training_labels = np.zeros(shape, dtype=np.uint8)
    Xmin = np.min(coordinates[:, 0])
    Xmax = np.max(coordinates[:, 0])
    Ymin = np.min(coordinates[:, 1])
    Ymax = np.max(coordinates[:, 1])
    grid = hexagonal_grid(r, shape)
    grid = grid.T
    grid = grid[::every_x_spots, :]

    for coor in grid:
        training_labels[disk((coor[1], coor[0]), r)] = label

    for coor in coordinates.T:
        training_labels[disk((coor[1], coor[0]), r * 4)] = 0

    return training_labels


def hexagonal_grid(SpotSize, shape):
    """
    Generate a hexagonal grid.

    Parameters
    ----------
    SpotSize : float
        Size of the spots.
    shape : tuple
        Shape of the grid.

    Returns
    -------
    numpy.ndarray
        Array containing the coordinates of the grid.
    """

    helper = SpotSize
    X1 = np.linspace(helper, shape[0] - helper, round(shape[0] / helper))
    Y1 = np.linspace(helper, shape[1] - 2 * helper, round(shape[1] / (2 * helper)))
    X2 = X1 + SpotSize / 2
    Y2 = Y1 + helper
    Gx1, Gy1 = np.meshgrid(X1, Y1)
    Gx2, Gy2 = np.meshgrid(X2, Y2)
    positions1 = np.vstack([Gy1.ravel(), Gx1.ravel()])
    positions2 = np.vstack([Gy2.ravel(), Gx2.ravel()])
    positions = np.hstack([positions1, positions2])
    return positions


def overlay_labels(im1, im2, alpha=0.8, show=True):
    """
    Merge two images using alpha blending.

    Parameters
    ----------
    im1 : numpy.ndarray
        First image.
    im2 : numpy.ndarray
        Second image.
    alpha : float, optional
        Blending factor. Default is 0.8.
    show : bool, optional
        Whether to show the merged plot or not. Default is True.

    Returns
    -------
    numpy.ndarray
        Merged image.
    """

    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.dpi"] = 100

    out_img = np.zeros(im1.shape, dtype=im1.dtype)
    out_img[:, :, :] = (alpha * im1[:, :, :]) + ((1 - alpha) * im2[:, :, :])
    out_img[:, :, 3] = 255

    if show:
        plt.imshow(out_img, origin='lower')

    return out_img


def find_files(directory, query):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if query in file:
                return os.path.join(root, file)



def anno_transfer(df_spots, df_grid, ppm_spots, ppm_grid, plot=True, how='nearest', max_distance=10e10):
    """
    Maps tissue annotations to Visium spots according to the nearest neighbors.
    
    Parameters
    ----------
    df_spots : pandas.DataFrame
        Dataframe with Visium spot data.
    df_grid : pandas.DataFrame
        Dataframe with grid data.
    ppm_spots : float 
        pixels per micron of spots
    ppm_grid : float 
        pixels per micron of grid
    plot : bool, optional
        If true, plots the coordinates of the grid space and the spot space to make sure 
        they are aligned. Default is True.
    how : string, optinal
        This determines how the association between the 2 grids is made from the scipy.interpolate.griddata function. Default is 'nearest'
    max_distance : int
        maximal distance where points are not migrated 

    Returns
    -------
    df_spots : pandas.DataFrame
        Updated dataframe with Visium spot data.
    """
    import numpy as np
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    import matplotlib.pyplot as plt
    
    print('Make sure the coordinate systems are aligned, e.g., axes are not flipped.') 
    a = np.vstack([df_grid['x']/ppm_grid, df_grid['y']/ppm_grid])
    b = np.vstack([df_spots['x']/ppm_spots, df_spots['y']/ppm_spots])
    
    if plot:
        plt.figure(dpi=100, figsize=[10, 10])
        plt.title('Spot space')
        plt.plot(b[0], b[1], '.', markersize=1)
        plt.show()
        
        plt.figure(dpi=100, figsize=[10, 10])
        plt.plot(a[0], a[1], '.', markersize=1)
        plt.title('Morpho space')
        plt.show()

    # Create new DataFrame
    new_df_spots = df_spots[['x', 'y']].copy()
    
    annotations = df_grid.columns[~df_grid.columns.isin(['x', 'y'])]
    
    for k in annotations:
        print('Migrating morphology - ' + k + ' to target space.')
        
        # Interpolation
        new_df_spots[k] = griddata(points=a.T, values=df_grid[k], xi=b.T, method=how)
        
        # Create KDTree
        tree = cKDTree(a.T)
        
        # Query tree for nearest distance
        distances, _ = tree.query(b.T, distance_upper_bound=max_distance)
        
        # Mask df_spots where the distance is too high
        new_df_spots[k][distances==np.inf] = None
  
    return new_df_spots



def anno_to_grid(folder, file_name, spot_diameter, load_colors=False,null_number=1):
    """
    Load annotations and transform them into a spot grid.
    
    Parameters
    ----------
    folder : str
        Folder path for annotations.
    file_name : str
        Name for tif image and pickle without extensions.
    spot_diameter : float
        The diameter used for grid.
    load_colors : bool, optional
        If True, get original colors used for annotations. Default is False.
    null_numer : int
        value of the label image where no useful information is stored e.g. background or unassigned pixels (usually 0 or 1). Default is 1

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with the grid annotations.
    """
    
    im, anno_order, ppm, anno_color = load_annotation(folder, file_name, load_colors)

    df = grid_anno(
        im,
        [im],
        [file_name],
        [anno_order],
        spot_diameter,
        ppm,
        ppm,
    )

    return df,ppm


def map_annotations_to_visium(vis_path, df_grid, ppm_grid, spot_diameter, plot=True, how='nearest', max_distance_factor=50, use_resolution='hires', res_in_ppm=1, count_file='raw_feature_bc_matrix.h5'):
    """
    Processes Visium data with high-resolution grid.

    Parameters
    ----------
    vis_path : str
        Path to the Visium data.
    df_grid : pandas.DataFrame
        Dataframe with grid data.
    ppm_grid : float 
        Pixels per micron of grid.
    spot_diameter : float
        Diameter of the spots.
    plot : bool, optional
        If true, plots the coordinates of the grid space and the spot space to make sure they are aligned. Default is True.
    how : string, optinal
        This determines how the association between the 2 grids is made from the scipy.interpolate.griddata function. Default is 'nearest'
    max_distance_factor : int
        Factor to calculate maximal distance where points are not migrated. The final max_distance used will be max_distance_factor * ppm_visium.
    use_resolution : str, optional
        Resolution to use. Default is 'hires'.
    res_in_ppm : float, optional
        Resolution in pixels per micron. Default is 1.
    count_file : str, optional
        Filename of the count file. Default is 'raw_feature_bc_matrix.h5'.

    Returns
    -------
    adata_vis : anndata.AnnData
        Annotated data matrix with Visium spot data and additional annotations.
    """
    import scanpy as sc
    # calculate distance matrix between hires and visium spots
    im, ppm_visium, visium_positions = read_visium(spaceranger_dir_path=vis_path+'/', use_resolution=use_resolution, res_in_ppm=res_in_ppm, plot=False)
    
    # rename columns for visium_positions DataFrame
    visium_positions.rename(columns={'pxl_row_in_fullres': "y", 'pxl_col_in_fullres': "x"}, inplace=True) 

    # Transfer annotations
    spot_annotations = anno_transfer(df_spots=visium_positions, df_grid=df_grid, ppm_spots=ppm_visium, ppm_grid=ppm_grid, plot=plot, how=how, max_distance=max_distance_factor * ppm_visium)

    # Read visium data
    adata_vis = sc.read_visium(vis_path, count_file=count_file)

    # Merge with adata_vis
    adata_vis.obs = pd.concat([adata_vis.obs, spot_annotations], axis=1)

    # Convert to int
    adata_vis.obsm['spatial'] = adata_vis.obsm['spatial'].astype('int')

    # Add to uns
    adata_vis.uns['hires_grid'] = df_grid
    adata_vis.uns['hires_grid_ppm'] = ppm_grid
    adata_vis.uns['hires_grid_diam'] = spot_diameter
    adata_vis.uns['visium_ppm'] = ppm_visium



    return adata_vis


def load_and_combine_annotations(folder, file_names, spot_diameter, load_colors=True):
    """
    Load tissue annotations from multiple files and combine them into a single DataFrame.

    Parameters
    ----------
    folder : str
        Folder path where the annotation files are stored.
    file_names : list of str
        List of names of the annotation files.
    spot_diameter : int
        Diameter of the spots.
    load_colors : bool, optional
        Whether to load colors. Default is True.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame that combines all the loaded annotations.
    ppm_grid : float
        Pixels per micron for the grid of the last loaded annotation.
    """
    df_list = []
    ppm_grid = None

    for file_name in file_names:
        df, ppm_grid = anno_to_grid(folder=folder, file_name=file_name, spot_diameter=spot_diameter, load_colors=load_colors)
        df_list.append(df)

    # Concatenate all dataframes
    df = pd.concat(df_list, join='inner', axis=1)

    # Remove duplicated columns
    df = df.loc[:, ~df.columns.duplicated()].copy()

    return df, ppm_grid

def annotate_l2(df_target, df_grid, annotation='annotations_level_0', KNN=10, max_distance_factor=50, plot=False,calc_dist=True):
    """
    Process the given AnnData object to calculate distances and perform annotation transfer.
    
    Parameters
    ----------
    df_target : pandas.DataFrame
        Dataframe with target data to be annotated.
    df_grid : pandas.DataFrame
        Dataframe with annotation data.
    annotation : str, optional
        Annotation column to be used for calculating distances, by default 'annotations_level_0'.
    ppm_grid : float
        should be the ppm resolution fot the grid data for visium would be stored here - adata_vis.uns['hires_grid_ppm']
    ppm_spots : float
        should be the ppm resolution fot the target data for visium would be stored here - adata_vis.uns['visium_ppm']
    KNN : int, optional
        Number of nearest neighbors to be considered for distance calculation, by default 10.
    max_distance_factor : int, optional
        Factor by which to calculate the maximum distance for annotation transfer, by default 50 in microns.
    plot : bool, optional
        Whether to plot during the annotation transfer, by default False.
     calc_dist : bool, optional
        If true, calculates the L2 distance. Default is True otherwise just migrates discrete annotations.
    
    Returns
    -------
    anndata.AnnData
        Processed AnnData object with updated observations.
    """
    
    df = df.obs[['x','y']].dropna()
    dist2cluster_fast(df_grid, annotation=annotation, KNN=KNN,calc_dist=calc_dist) # calculate minimum mean distance of each spot to clusters 
    df_grid_new = df_grid.filter(like=annotation)
    df_grid_new['x'] = df_grid['x']
    df_grid_new['y'] = df_grid['y']
    spot_annotations = anno_transfer(df_spots=df_visium, df_grid=df_grid_new, ppm_spots=ppm_spots, ppm_grid=adata_vis.uns['hires_grid_ppm'], max_distance=max_distance_factor * adata_vis.uns['visium_ppm'], plot=plot)
    for col in spot_annotations:
        adata_vis.obs[col] = spot_annotations[col]
        adata_vis.uns['hires_grid'][col] = df_grid_new[col]
    # df1 = pd.concat([adata_vis.obs, spot_annotations], axis=1)
    # df1 = df1.loc[:, ~df1.columns.duplicated()].copy()
    # adata_vis.obs = df1

    return adata_vis



def map_annotations_to_target(df_source, df_target, ppm_source,ppm_target, plot=True, how='nearest', max_distance=50):
    """
    map annotations to any form of of csv where you have x y cooodinates spot df (cells or spots) data with high-resolution grid.
    note! - xy coordinates must be named 'x' and 'y'

    Parameters
    ----------
    df_source : pandas.DataFrame
        Dataframe with grid data.
    df_target : pandas.DataFrame
        Dataframe with target data.
    ppm_source : float 
        Pixels per micron of source data.
    ppm_target : float 
        Pixels per micron of target data.
    plot : bool, optional
        If true, plots the coordinates of the grid space and the spot space to make sure they are aligned. Default is True.
    how : string, optinal
        This determines how the association between the 2 grids is made from the scipy.interpolate.griddata function. Default is 'nearest'
        if the data is categorial then only the 'nearest' would work but if interpolation is needed one should supbset to only numeric data.
    max_distance : int
        Factor to calculate maximal distance where points are not migrated. The final max_distance used will be max_distance * ppm_target.
   
    Returns
    -------
    df_target : pandas.DataFrame
        Annotated dataframe with additional annotations from the source data.
    """
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    
    # generate matched coordinate space 
    a = np.vstack([df_source['x']/ppm_source, df_source['y']/ppm_source])
    b = np.vstack([df_target['x']/ppm_target, df_target['y']/ppm_target])
    
    if plot:
        print('Make sure the coordinate systems are aligned, e.g., axes are not flipped and the resolution is matched.') 
        plt.figure(dpi=100, figsize=[10, 10])
        plt.title('Target space')
        plt.plot(b[0], b[1], '.', markersize=1)
        plt.show()
        
        plt.figure(dpi=100, figsize=[10, 10])
        plt.plot(a[0], a[1], '.', markersize=1)
        plt.title('source space')
        plt.show()

    annotations = df_source.columns[~df_source.columns.isin(['x', 'y'])] # extract annotation categories
    
    for k in annotations:
        print('Migrating source annotation - ' + k + ' to target space.')
        
        # Interpolation
        df_target[k] = griddata(points=a.T, values=df_source[k], xi=b.T, method=how)
        
        # Create KDTree
        tree = cKDTree(a.T)
        
        # Query tree for nearest distance
        distances, _ = tree.query(b.T, distance_upper_bound=max_distance)
        
        # Mask df_spots where the distance is too high
        df_target[k][distances==np.inf] = None
  
    return df_target


def read_visium_table(vis_path):
    """
    This function reads a scale factor from a json file and a table from a csv file, 
    then calculates the 'ppm' value and returns the table with the new column names.
    
    note that for CytAssist the header is changes so this funciton should be updated
    """
    with open(vis_path + '/spatial/scalefactors_json.json', 'r') as f:
        scalef = json.load(f)

    ppm = scalef['spot_diameter_fullres'] / 55 

    df_visium_spot = pd.read_csv(vis_path + '/spatial/tissue_positions_list.csv', header=None)

    df_visium_spot.rename(columns={4:'y',5:'x',1:'in_tissue',0:'barcode'}, inplace=True)
    df_visium_spot.set_index('barcode', inplace=True)

    return df_visium_spot, ppm

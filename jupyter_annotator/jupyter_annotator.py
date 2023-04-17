import base64
import json
import pickle
import random
import warnings
from functools import partial
from io import BytesIO

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import skimage
from bokeh.models import (FreehandDrawTool, PolyDrawTool, PolyEditTool,
                          TabPanel, Tabs)
from bokeh.plotting import figure, show
from PIL import (Image, ImageColor, ImageDraw, ImageEnhance, ImageFilter,
                 ImageFont)
from scipy import interpolate
from scipy.spatial import distance
from skimage import data, feature, future, segmentation
from skimage.draw import polygon, disk
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from packaging import version
import bokeh

try:
    import scanpy as sc
except:
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
    ImageDraw.Draw(img).text((5
                              , 2), name,fill=tuple((np.array(matplotlib.colors.to_rgb(color))*255).astype(int)),
                              font=ImageFont.truetype(font_path, font_size))
    if version.parse(bokeh.__version__) < version.parse("3.1.0"):
        img = to_base64(img)
    return img


def read_image(
    path,
    ppm=None,
    scale=1,
    scaleto1ppm=True,
    contrast_factor=1,
    background_image_path=None,
  
):
    """
        Read H&E image 
        Parameters
        ----------     
        path 
            path to image, must follow supported Pillow formats - https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
        categorical_covariate_keys
        scale 
            a factor to scale down image, if this is applied (any value smaller than 1) a gaussian filter would be applied 

    """
    
    im = Image.open(path)
    
    if scale<1:
        width, height = im.size
        newsize = (int(width*scale), int(height*scale))
        im = im.resize(newsize,Image.Resampling.LANCZOS)
        if ppm:
            ppm_out = ppm*scale
            
    if scaleto1ppm:
        if not(ppm):
            try:
                ppm = im.info['resolution'][0]
            except:
                print('scale to 1 ppm selected, please provide ppm' )
        ppm_out = 1
        width, height = im.size
        newsize = (int(width/ppm), int(height/ppm))
        im = im.resize(newsize,Image.Resampling.LANCZOS)

    im = im.convert("RGBA")
    enhancer = ImageEnhance.Contrast(im)
    factor = contrast_factor #increase contrast
    im = enhancer.enhance(factor*factor)
    
    if background_image_path:
        im2 = Image.open(background_image_path)
        im2 = im2.convert("RGBA")
        enhancer = ImageEnhance.Contrast(im2)
        factor = contrast_factor #increase contrast
        im2 = enhancer.enhance(factor*factor)
        im2 = im2.resize(newsize,Image.Resampling.LANCZOS)
        im2 = im2.convert("RGBA")
        im = simonson_vHE(np.array(im).astype('uint8'),np.array(im2).astype('uint8'))
        
    return np.array(im), ppm_out

def read_visium(
    SpaceRanger_dir_path,
    use_resolution='hires',
    res_in_ppm = None,
    fullres_path = None,  
):
    """
        Read 10X visium image data from spaceranger (1.3.0)
        
        Parameters
        ----------     
        SpaceRanger_dir_path 
            path to 10X SpaceRanger output folder
        use_resolution 
            image resolution to use either 'hires', 'lowres' or 'fullres'
                            'fullres' is the image that was sent to SpaceRanger (togehter with sequencing data)
                            if user_resolution == 'fullres' then fullres_path need to be specified
        fullres_path
            path to fullres image used for mapping 
        res_in_ppm
            when using full resolution used to resize the full image to 0.5 pixels per microns unless stated eitherwise

    """
    spotsize = 55 #um

    scalef = json.load(open(SpaceRanger_dir_path+'spatial/scalefactors_json.json','r'))
    if use_resolution=='fullres':
        assert fullres_path is not None, 'if use_resolution=="fullres" fullres_path has to be specified'

    df = pd.read_csv(SpaceRanger_dir_path+'spatial/tissue_positions_list.csv',header=None)
    df = df.set_index(keys=0)
    df = df[df[1]>0] # in tissue
    
    # turn df to mu
    fullres_ppm = scalef['spot_diameter_fullres']/spotsize
    df[4] = df[4]/fullres_ppm
    df[5] = df[5]/fullres_ppm
    
    if use_resolution=='fullres':
        im = Image.open(fullres_path)
        ppm = fullres_ppm
    else:
        im = Image.open(SpaceRanger_dir_path+'spatial/tissue_'+use_resolution+'_image.png')
        ppm = scalef['spot_diameter_fullres']*scalef['tissue_'+use_resolution+'_scalef']/spotsize
        
    
    if res_in_ppm:
        width, height = im.size
        newsize = (int(width*res_in_ppm/ppm), int(height*res_in_ppm/ppm))
        im = im.resize(newsize,Image.Resampling.LANCZOS)
        ppm = res_in_ppm
    
    # translate from mu to pixel
    df[4] = df[4]*ppm
    df[5] = df[5]*ppm
    

    im = im.convert("RGBA")
    return np.array(im), ppm, df


def scribbler(
    imarray,
    anno_dict,
    plot_scale,
):
    """
        interactive scribble line annotations with Bokeh  
        
        Parameters
        ----------     
        imarray  
            image in numpy array format (nparray)
        anno_dict
            dictionary of structures to annotate and colors for the structures     

    """

    imarray_c = imarray.astype('uint8')[:,:].copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])

    p =  figure(width=int(imarray_c.shape[1]/3.5*plot_scale),height=int(imarray_c.shape[0]/3.5*plot_scale),match_aspect=True)
    plotted_image = p.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    anno_color_map = anno_dict
    anno_color_map
    render_dict = {}
    draw_tool_dict = {}
    for l in list(anno_dict.keys()):
        render_dict[l] = p.multi_line([], [], line_width=5, alpha=0.4, color=anno_color_map[l])
        draw_tool_dict[l] = FreehandDrawTool(renderers=[render_dict[l]], num_objects=200, icon=create_icon(l[0],anno_color_map[l]))
        draw_tool_dict[l].description = l
        p.add_tools(draw_tool_dict[l])
    
    
    return p, render_dict


def annotator(
    imarray,
    annotation,
    anno_dict,
    fig_downsize_factor = 5,
    
):
    """
        interactive annotation tool with line annotations using Bokeh tabs for toggling between morphology and annotation. 
        The principle is that selecting closed/semiclosed shaped that will later be filled accordind to the proper annotation.
        
        Parameters
        ----------     
        imarray  
            image in numpy array format (nparray)
        annotation  
            label image in numpy array format (nparray)
        anno_dict
            dictionary of structures to annotate and colors for the structures             
        fig_downsize_factor
            a plotting thing

    """
    
    # tab1
    imarray_c = annotation[:,:].copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])
    # p = figure(width=int(imarray_c.shape[1]/fig_downsize_factor),height=int(imarray_c.shape[0]/fig_downsize_factor))
    p = figure(width=int(imarray_c.shape[1]/fig_downsize_factor),height=int(imarray_c.shape[0]/fig_downsize_factor),match_aspect=True)
    plotted_image = p.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    tab1 = TabPanel(child=p, title="Annotation")

    # tab2
    imarray_c = imarray[:,:].copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])
    p1 = figure(width=int(imarray_c.shape[1]/fig_downsize_factor),height=int(imarray_c.shape[0]/fig_downsize_factor),match_aspect=True, x_range=p.x_range,y_range=p.y_range)
    plotted_image = p1.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    tab2 = TabPanel(child=p1, title="Image")

    # # tab3
    # imarray_c = result_rgb[:,:].copy()
    # np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])
    # p2 = figure(width=int(imarray_c.shape[1]/fig_downsize_factor),height=int(imarray_c.shape[0]/fig_downsize_factor), x_range=p.x_range,y_range=p.y_range)
    # plotted_image = p2.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    # tab3 = TabPanel(child=p2, title="Annotation")
    anno_color_map = anno_dict
    anno_color_map


    from bokeh.models import ColumnDataSource
    from bokeh.core.properties import field

    

    # brushes
    render_dict = {}
    draw_tool_dict = {}
    source_data_dict = {}

    render_dict_2 = {}
    draw_tool_dict_2 = {}

    for l in list(anno_dict.keys()):
        draw_source = ColumnDataSource(data=dict(xs=[], ys=[]))

        render_dict[l] = p.multi_line(field("xs"), field("ys"), line_width=3, alpha=0.4, color=anno_color_map[l], source=draw_source)
        draw_tool_dict[l] = FreehandDrawTool(renderers=[render_dict[l]], num_objects=100, icon=create_icon(l[0],anno_color_map[l]))
        draw_tool_dict[l].description = l
        p.add_tools(draw_tool_dict[l])

        render_dict_2[l] = p1.multi_line(field("xs"), field("ys"), line_width=3, alpha=0.4, color=anno_color_map[l], source=draw_source)
        draw_tool_dict_2[l] = FreehandDrawTool(renderers=[render_dict_2[l]], num_objects=100, icon=create_icon(l[0],anno_color_map[l]))
        draw_tool_dict_2[l].description = l
        p1.add_tools(draw_tool_dict_2[l])

        source_data_dict[l] = draw_source

    tabs = Tabs(tabs=[tab1, tab2])
    return tabs, render_dict

def complete_pixel_gaps(x,y):
    
    newx1 = []
    newx2 = []
    for idx,px in enumerate(x[:-1]):
        f = interpolate.interp1d(x[idx:idx+2], y[idx:idx+2])
        gapx1 = np.linspace(x[idx],x[idx+1],num=np.abs(x[idx+1]-x[idx]+1))
        gapx2 = f(gapx1).astype(int)
        newx1 = newx1 + list(gapx1[:]) 
        newx2 = newx2 + list(gapx2[:]) 

    newy1 = []
    newy2 = []
    for idx,py in enumerate(y[:-1]):
        f = interpolate.interp1d(y[idx:idx+2], x[idx:idx+2])
        gapy1 = np.linspace(y[idx],y[idx+1],num=np.abs(y[idx+1]-y[idx]+1))
        gapy2 = f(gapy1).astype(int)
        newy1 = newy1 + list(gapy1[:]) 
        newy2 = newy2 + list(gapy2[:]) 
    newx = newx1 + newy2
    newy = newx2 + newy1


    return newx,newy


def scribble_to_labels(
    imarray,
    render_dict,
    line_width = 10,
):
    """
        extract scribbles to a label image 
        
        Parameters
        ----------     
        imarray  
            image in numpy array format (nparray) used to calculate the label image size
        render_dict
            Bokeh object carrying annotations 
        line_width
            width of the line labels (int)

    """
    
    annotations = {}
    training_labels = np.zeros((imarray.shape[1],imarray.shape[0]), dtype=np.uint8) # blank annotation image
    # annotations = pd.DataFrame()
    for idx,a in enumerate(render_dict.keys()):
        print(a)
        xs = []
        ys = []
        annotations[a] = []
        for o in range(len(render_dict[a].data_source.data['xs'])):
            xt,yt = complete_pixel_gaps(np.array(render_dict[a].data_source.data['xs'][o]).astype(int),np.array(render_dict[a].data_source.data['ys'][o]).astype(int))
            xs = xs + xt
            ys = ys + yt
            annotations[a] = annotations[a] + [np.vstack([np.array(render_dict[a].data_source.data['xs'][o]).astype(int),np.array(render_dict[a].data_source.data['ys'][o]).astype(int)])] # to save 

        training_labels[np.floor(xs).astype(int),np.floor(ys).astype(int)] = idx+1
        # df = pd.DataFrame(render_dict[a].data_source.data)
        # df.index = a+'-'+df.index.astype('str')
        # annotations = pd.concat([annotations,df])
    training_labels = training_labels.transpose()
    import skimage as sk 
    return sk.segmentation.expand_labels(training_labels, distance=line_width/2)


def rgb_from_labels(labelimage,colors):

    labelimage_rgb = np.zeros((labelimage.shape[0],labelimage.shape[1] ,4))
    
    for c in range(len(colors)):
        color = ImageColor.getcolor(colors[c], "RGB")
        labelimage_rgb[np.where(labelimage == c+1)[0],np.where(labelimage == c+1)[1],0:3] = np.array(color)
    labelimage_rgb[:,:,3] = 255
    return labelimage_rgb.astype('uint8')


def sk_rf_classifier(
    im,
    training_labels
    
):


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


def overlay_lebels(im1,im2,alpha=0.8,show=True):
    #generate overlay image
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.dpi"] = 100
    out_img = np.zeros(im1.shape,dtype=im1.dtype)
    out_img[:,:,:] = (alpha * im1[:,:,:]) + ((1-alpha) * im2[:,:,:])
    out_img[:,:,3] = 255
    if show:
        plt.imshow(out_img,origin='lower')
    return out_img
   
def update_annotator(
    imarray,
    result,
    anno_dict,
    render_dict,
    alpha,
):
    """
        updates annotations and generates overly (out_img) and the label image (corrected_labels)
        
        Parameters
        ----------     
        imarray  
            image in numpy array format (nparray)
        result  
            label image in numpy array format (nparray)
        anno_dict
            dictionary of structures to annotate and colors for the structures     
        render_dict
            bokeh data container

    """
    
    
    corrected_labels = result.copy()
    # annotations = pd.DataFrame()
    for idx,a in enumerate(render_dict.keys()):
        if render_dict[a].data_source.data['xs']:
            print(a)
            for o in range(len(render_dict[a].data_source.data['xs'])):
                x = np.array(render_dict[a].data_source.data['xs'][o]).astype(int)
                y = np.array(render_dict[a].data_source.data['ys'][o]).astype(int)
                rr, cc = polygon(y, x)
                inshape = np.where(np.array(result.shape[0]>rr) & np.array(0<rr) & np.array(result.shape[1]>cc) & np.array(0<cc))[0]
                corrected_labels[rr[inshape], cc[inshape]] = idx+1 
                # make sure pixels outside the image are ignored

    #generate overlay image
    rgb = rgb_from_labels(corrected_labels,list(anno_dict.values()))
    out_img = overlay_lebels(imarray,rgb,alpha=alpha,show=False)
    # out_img = out_img.transpose() 
    return out_img, corrected_labels


def rescale_image(
    label_image,
    target_size,
):
    """
        rescales label image to original image size 
        
        Parameters
        ----------     
        label_image  
            labeled image (nparray)
        scale  
            factor to enlarge image

    """
    imP = Image.fromarray(label_image)
    newsize = (target_size[0], target_size[1])
    
    return np.array(imP.resize(newsize))


def save_annotation(
    folder,
    label_image,
    file_name, 
    anno_names,
    anno_colors,
    ppm
):
    """
        saves the annotated image as .tif and in addition saves the translation from annotations to labels in a pickle file 
        
        Parameters
        ----------     
        label_image  
            labeled image (nparray)
        file_name  
            name for tif image and pickle

    """

    label_image = Image.fromarray(label_image)
    label_image.save(folder+file_name+'.tif')
    with open(folder+file_name+'.pickle', 'wb') as handle:
        pickle.dump(dict(zip(range(1,len(anno_names)+1),anno_names)), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+file_name+'colors.pickle', 'wb') as handle:
        pickle.dump(dict(zip(anno_names,anno_colors)), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+'ppm.pickle', 'wb') as handle:
        pickle.dump({'ppm':ppm}, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_annotation(
    folder,
    file_name, 
    load_colors = False,
):
    """
        saves the annotated image as .tif and in addition saves the translation from annotations to labels in a pickle file 
        
        Parameters
        ----------     
        folder            
            Folder path for annotations
        file_name  
            name for tif image and pickle without extensions

    """
    imP = Image.open(folder+file_name+'.tif')
    
    ppm = imP.info['resolution'][0]
    im = np.array(imP)
    

    print('loaded annotation image - '+file_name+' size - '+str(im.shape))
    with open(folder+file_name+'.pickle', 'rb') as handle:
        anno_order = pickle.load(handle)
        print('loaded annotations')        
        print(anno_order)
    with open(folder+'ppm.pickle', 'rb') as handle:
        ppm = pickle.load(handle)
        print('loaded ppm')        
        print(ppm)
        
    if load_colors:
        with open(folder+file_name+'colors.pickle', 'rb') as handle:
            anno_color = pickle.load(handle)
            print('loaded color annotations')        
            print(anno_color)
        return im, anno_order, ppm['ppm'], anno_color
    
    else:
        return im, anno_order, ppm['ppm']
    
    

#The following notebook is a series of simple examples of applying the method to data on a 
#CODEX/Keyence microscrope to produce virtual H&E images using fluorescence data.  If you 
#find it useful, will you please consider citing the relevant article?:

#Creating virtual H&E images using samples imaged on a commercial CODEX platform
#Paul D. Simonson, Xiaobing Ren,  Jonathan R. Fromm
#doi: https://doi.org/10.1101/2021.02.05.21249150
#Submitted to Journal of Pathology Informatics, December 2020
def simonson_vHE(
    dapi_image,
    eosin_image,
):
    def createVirtualHE(dapi_image, eosin_image, k1, k2, background, beta_DAPI, beta_eosin):
        new_image = np.empty([dapi_image.shape[0], dapi_image.shape[1], 4])
        new_image[:,:,0] = background[0] + (1 - background[0]) * np.exp(- k1 * beta_DAPI[0] * dapi_image - k2 * beta_eosin[0] * eosin_image)
        new_image[:,:,1] = background[1] + (1 - background[1]) * np.exp(- k1 * beta_DAPI[1] * dapi_image - k2 * beta_eosin[1] * eosin_image)
        new_image[:,:,2] = background[2] + (1 - background[2]) * np.exp(- k1 * beta_DAPI[2] * dapi_image - k2 * beta_eosin[2] * eosin_image)
        new_image[:,:,3] = 1
        new_image = new_image*255
        return new_image.astype('uint8')

    #Defaults:
    k1 = k2 = 0.001

    background_red = 0.25
    background_green = 0.25
    background_blue = 0.25
    background = [background_red, background_green, background_blue]

    beta_DAPI_red = 9.147
    beta_DAPI_green = 6.9215
    beta_DAPI_blue = 1.0
    beta_DAPI = [beta_DAPI_red, beta_DAPI_green, beta_DAPI_blue]

    beta_eosin_red = 0.1
    beta_eosin_green = 15.8
    beta_eosin_blue = 0.3
    beta_eosin = [beta_eosin_red, beta_eosin_green, beta_eosin_blue]


    dapi_image = dapi_image[:,:,0]+dapi_image[:,:,1]
    eosin_image = eosin_image[:,:,0]+eosin_image[:,:,1]

    print(dapi_image.shape)
    return createVirtualHE(dapi_image, eosin_image, k1, k2, background=background, beta_DAPI=beta_DAPI, beta_eosin=beta_eosin)



def generate_hires_grid(
    im,
    spot_diameter,
    pixels_per_micron,
):
    
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
    pixels_per_micron,
    
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
        positions 
            grid positions

    """

    print('generating grid with spot size - '+str(spot_diameter)+', with resolution of - '+str(pixels_per_micron)+' ppm')
    positions = generate_hires_grid(im,spot_diameter,pixels_per_micron)
    positions = positions.astype('float32')
    dim = [im.shape[0],im.shape[1]]
    # transform tissue annotation images to original size

    radius = spot_diameter/8
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
            disk = skimage.draw.disk([int(pointcell[1]),int(pointcell[0])],radius)
            anno_dict[idx1] = annotation_label_list[idx0][int(np.median(anno_orig[disk]))]
            number_dict[idx1] = int(np.median(anno_orig[disk]))
        df[annotation_image_names[idx0]] = anno_dict.values()
        df[annotation_image_names[idx0]+'_number'] = number_dict.values()
    df['index'] = df['index'].astype(int)
    df.set_index('index', drop=True, inplace=True)
    return df



# measure for each spot the mean closest distance to the x closest spoint of a given structure, resolution is x 
def dist2cluster(
    df,
    annotation,
    distM,
    resolution=4,
    calc_dist=True,
    logscale = False
):
    Dist2ClusterAll = {}    
    categories = np.unique(df[annotation])
    for idx, c in enumerate(categories): # measure edistange to all
        indextmp = df[annotation]==c
        if len(np.where(indextmp)[0])>resolution: # this is an important point to see if there are enough points from a structure to calculate mean distance 
            print(c)
            Dist2ClusterAll[c] =  np.median(np.sort(distM[indextmp,:],axis=0)[range(resolution),:],axis=0) # was 12

    # update annotations in AnnData 
    for c in categories: 
        if c!='unassigned':
            if calc_dist:
                if logscale:
                    df["L2_dist_log10_"+annotation+'_'+c] = np.log10(Dist2ClusterAll[c])
                else:
                    df["L2_dist_"+annotation+'_'+c] = Dist2ClusterAll[c]
            df[annotation] = categories[np.argmin(np.array(list(Dist2ClusterAll.values())),axis=0)]
    return Dist2ClusterAll


def axis_2p_norm(
    Dist2ClusterAll,
    structure_list,
    weights = [1,1], 
):
    
    warnings.filterwarnings("ignore")
    # CMA calculations 
    fa = weights[0]
    fb = weights[0]
    axis = np.array([( fa*int(a) - fb*int(b) ) / ( fa*int(a) + fb*int(b) ) for a,b in zip(Dist2ClusterAll[structure_list[0]], Dist2ClusterAll[structure_list[1]])])
    return axis

def bin_axis(
    ct_order,
    cutoff_vales,
    df,
    axis_anno_name,
):

    # # manual annotations
    df['manual_bin_'+axis_anno_name] = 'unassigned'
    df['manual_bin_'+axis_anno_name] = df['manual_bin_'+axis_anno_name].astype('object')
    df.loc[np.array(df[axis_anno_name]<cutoff_vales[0]) ,'manual_bin_'+axis_anno_name] = ct_order[0]
    for idx,r in enumerate(cutoff_vales[:-1]):
        print(ct_order[idx+1])
        print(str(cutoff_vales[idx])+','+str(cutoff_vales[idx+1]))
        df.loc[np.array(df[axis_anno_name]>=cutoff_vales[idx]) & np.array(df[axis_anno_name]<cutoff_vales[idx+1]),'manual_bin_'+axis_anno_name] = ct_order[idx+1]

    df.loc[np.array(df[axis_anno_name]>=cutoff_vales[-1]),'manual_bin_'+axis_anno_name] = ct_order[-1]
    df['manual_bin_'+axis_anno_name] = df['manual_bin_'+axis_anno_name].astype('category')
    df['manual_bin_'+axis_anno_name+'_int'] =  df['manual_bin_'+axis_anno_name].cat.codes
   
    return df


def axis_3p_norm(
    Dist2ClusterAll,
    structure_list,
    df,
    axis_anno='cma_v3',
):
    df[axis_anno] = np.zeros(df.shape[0])
    counter = -1
    for b,c,a in zip(Dist2ClusterAll[structure_list[0]], Dist2ClusterAll[structure_list[1]],Dist2ClusterAll[structure_list[2]]):
        counter = counter+1
        if (b>=c): # meaning you are in the medulla 
            df[axis_anno].iloc[counter] = (int(b)-int(c))/(int(c)+int(b)) # non linear normalized distance between cortex with medulla for edge effect modulation 
        if (b<c): # meaning you are in the cortex 
            df[axis_anno].iloc[counter] = (int(a)-int(c)+0.5*int(b))/(int(a)+int(c)+0.5*int(b))-1 # shifted non linear distance between edge and medualla 
    return df 

    
def anno_to_cells(
    df_cells,
    df_morphology,
    numerical_annotations, 
    categorical_annotation_names, 
    categorical_annotation_number_names, 
):
    
    print('make sure the coordinate systems are alligned e.g. axes are not flipped') 
    a = np.vstack([df_morphology['x'],df_morphology['y']])
    b = np.vstack([df_cells['centroid-1'],df_cells['centroid-0']])
    # xi = np.vstack([dfseg['centroid-1'],dfseg['centroid-0']]).T
    plt.figure(dpi=100, figsize=[10,10])
    plt.title('cell space')
    plt.plot(b[0],b[1],'.', markersize=1)
    plt.show()
    plt.figure(dpi=100, figsize=[10,10])
    plt.plot(a[0],a[1],'.', markersize=1)
    plt.title('morpho spcae')
    plt.show()


    # migrate continues annotations
    xi = np.vstack([df_cells['centroid-1'],df_cells['centroid-0']]).T
    for k in numerical_annotations:
        print('migrating - '+k+' to segmentations')
        df_cells[k] = scipy.interpolate.griddata(points=a.T, values = df_morphology[k], xi=b.T,method='cubic')
        # plt.title(k)
        # df_cells[k].hist(bins=5)
        # plt.show()

    # migrate categorial annotations
    for idx,k in enumerate(categorical_annotation_names):
        df_cells[categorical_annotation_number_names[idx]] = scipy.interpolate.griddata(points=a.T, values = df_morphology[categorical_annotation_number_names[idx]], xi=b.T,method='nearest')
        dict_temp = dict(zip(df_morphology[categorical_annotation_number_names[idx]].value_counts().keys(),df_morphology[k].value_counts().keys()))
        print('migrating - '+k+' to segmentations')
        df_cells[k] = df_cells[categorical_annotation_number_names[idx]].map(dict_temp)
        print(df_cells[k].value_counts())
        
    return df_cells



def anno_to_visium_spots(
    df_vis,
    df_morphology,
    numerical_annotations, 
    categorical_annotation_names, 
    categorical_annotation_number_names, 
):
    
    print('make sure the coordinate systems are alligned e.g. axes are not flipped') 
    a = np.vstack([df_morphology['x'],df_morphology['y']])
    b = np.vstack([df_vis[5],df_vis[4]])
    # xi = np.vstack([dfseg['centroid-1'],dfseg['centroid-0']]).T
    plt.figure(dpi=100, figsize=[10,10])
    plt.title('visium space')
    plt.plot(b[0],b[1],'.', markersize=1)
    plt.show()
    plt.figure(dpi=100, figsize=[10,10])
    plt.plot(a[0],a[1],'.', markersize=1)
    plt.title('morpho spcae')
    plt.show()



    # migrate continues annotations
    xi = np.vstack([df_vis[5],df_vis[4]]).T
    for k in numerical_annotations:
        print('migrating - '+k+' to spots')
        df_vis[k] = scipy.interpolate.griddata(points=a.T, values = df_morphology[k], xi=b.T,method='cubic')
        # plt.title(k)
        # df_vis[k].hist(bins=5)
        # plt.show()

    # migrate categorial annotations
    for idx,k in enumerate(categorical_annotation_names):
        df_vis[categorical_annotation_number_names[idx]] = scipy.interpolate.griddata(points=a.T, values = df_morphology[categorical_annotation_number_names[idx]], xi=b.T,method='nearest')
        dict_temp = dict(zip(df_morphology[categorical_annotation_number_names[idx]].value_counts().keys(),df_morphology[k].value_counts().keys()))
        print('migrating - '+k+' to spots')
        df_vis[k] = df_vis[categorical_annotation_number_names[idx]].map(dict_temp)
        print(df_vis[k].value_counts())
        
    return df_vis


def plot_grid(
    df,
    annotation,
    spotsize=10,
    save=False,
    dpi=100,
    figsize=[5,5],
    savepath = None,
    
):   
    
    plt.figure(dpi=dpi, figsize=figsize)
    
    ct_order = list((df[annotation].value_counts()>0).keys())
    ct_color_map = dict(zip(ct_order, np.array(sns.color_palette("colorblind", len(ct_order)))[range(len(ct_order))]))
    sns.scatterplot(x='x',y='y',hue=annotation,s=spotsize,data = df,palette=ct_color_map,hue_order=ct_order)
    plt.grid(False)
    plt.title(annotation)
    plt.axis('equal')
    if save:
        plt.savefig(savepath+'/'+tr(annotation)+'.pdf')  
    plt.show()

    
    
    
    
def poly_annotator(
    imarray,
    annotation,
    anno_dict,
    fig_downsize_factor = 5,
    
):
    """
        interactive annotation tool with line annotations using Bokeh tabs for toggling between morphology and annotation. 
        The principle is that selecting closed/semiclosed shaped that will later be filled accordind to the proper annotation.
        
        Parameters
        ----------     
        imarray  
            image in numpy array format (nparray)
        annotation  
            label image in numpy array format (nparray)
        anno_dict
            dictionary of structures to annotate and colors for the structures             
        fig_downsize_factor
            a plotting thing

    """
    


    # tab1
    imarray_c = annotation[:,:].copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])
    # p = figure(width=int(imarray_c.shape[1]/fig_downsize_factor),height=int(imarray_c.shape[0]/fig_downsize_factor))
    p = figure(width=int(imarray_c.shape[1]/fig_downsize_factor),height=int(imarray_c.shape[0]/fig_downsize_factor),match_aspect=True)
    plotted_image = p.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    tab1 = TabPanel(child=p, title="Annotation")

    # tab2
    imarray_c = imarray[:,:].copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])
    p1 = figure(width=int(imarray_c.shape[1]/fig_downsize_factor),height=int(imarray_c.shape[0]/fig_downsize_factor),match_aspect=True, x_range=p.x_range,y_range=p.y_range)
    plotted_image = p1.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    tab2 = TabPanel(child=p1, title="Image")


    anno_color_map = anno_dict
    anno_color_map

    # brushes
    render_dict = {}
    polt_tool_dict = {}
    for l in list(anno_dict.keys()):
        render_dict[l] = p.multi_line([], [], line_width=5, alpha=0.4, color=anno_color_map[l])
        polt_tool_dict[l] = PolyDrawTool(renderers=[render_dict[l]], num_objects=100, icon=create_icon(l[0],anno_color_map[l]))
        polt_tool_dict[l].description = l
        p.add_tools(polt_tool_dict[l])

    tabs = Tabs(tabs=[tab1, tab2])
    return tabs, render_dict

def object_annotator(
    imarray,
    result,
    anno_dict,
    render_dict,
    alpha,
    ):
    """
        extracts annotations and lables them according to bruch strokes while generating (out_img) and the label image (corrected_labels) and the anno_dict object 
        
        Parameters
        ----------     
        imarray  
            image in numpy array format (nparray)
        result  
            label image in numpy array format (nparray)
        anno_dict
            dictionary of structures to annotate and colors for the structures     
        render_dict
            bokeh data container

    """
    colorpool = ['yellow','green','cyan','brown','magenta','blue','red','orange']


    corrected_labels = result.copy()
    # annotations = pd.DataFrame()
    object_dict = {}
    for idx,a in enumerate(render_dict.keys()):
        if render_dict[a].data_source.data['xs']:
            print(a)
            for o in range(len(render_dict[a].data_source.data['xs'])):
                x = np.array(render_dict[a].data_source.data['xs'][o]).astype(int)
                y = np.array(render_dict[a].data_source.data['ys'][o]).astype(int)
                rr, cc = polygon(y, x)
                inshape = np.where(np.array(result.shape[0]>rr) & np.array(0<rr) & np.array(result.shape[1]>cc) & np.array(0<cc))[0]         # make sure pixels outside the image are ignored
                corrected_labels[rr[inshape], cc[inshape]] = o+1
                object_dict[a+'_'+str(o)] = random.choice(colorpool)

    #generate overlay image
    rgb = rgb_from_labels(corrected_labels,list(object_dict.values()))
    out_img = overlay_lebels(imarray,rgb,alpha=alpha,show=False)
    # out_img = out_img.transpose() 
    return out_img, corrected_labels, object_dict
  
  
def gene_labels(adata,df,training_labels,marker_dict,annodict,r,labels_per_marker):
    for m in list(marker_dict.keys()): 
        print(marker_dict[m])
        GeneIndex = np.where(adata.var_names.str.fullmatch(marker_dict[m]))[0]
        GeneData = adata.X[:,GeneIndex].todense()
        SortedExp = np.argsort(GeneData,axis=0)[::-1]
        list_gene = adata.obs.index[np.squeeze(SortedExp[range(labels_per_marker[m])])][0]
        for idx,sub in enumerate(list(annodict.keys())):
            if sub == m:
                back = idx
        for coor in df.loc[list_gene,4:5].to_numpy():
            training_labels[disk((coor[0],coor[1]),r/4)] = back+1
    return training_labels

def background_labels(shape,coordinates,r,every_x_spots=10,label=1):
    training_labels = np.zeros(shape, dtype=np.uint8)
    Xmin = np.min(coordinates[:,0])
    Xmax = np.max(coordinates[:,0])
    Ymin = np.min(coordinates[:,1])
    Ymax = np.max(coordinates[:,1])
    grid = hexagonal_grid(r,shape) # generate a grid over the entire image 
    grid = grid.T
    grid = grid[::every_x_spots,:]
    for coor in grid:
        training_labels[disk((coor[1],coor[0]),r)] = label
    for coor in coordinates.T:
        training_labels[disk((coor[1],coor[0]),r*4)] = 0
    
    # training_labels[int(Ymin):int(Ymax),int(Xmin):int(Xmax)] = 0 # remove spots from tisue area
    return training_labels

def hexagonal_grid(SpotSize,shape):
    helper = SpotSize
    X1 = np.linspace(helper,shape[0]-helper,round(shape[0]/helper))
    Y1 = np.linspace(helper,shape[1]-2*helper,round(shape[1]/(2*helper)))
    X2 = X1 + SpotSize/2
    Y2 = Y1 + helper
    Gx1, Gy1 = np.meshgrid(X1,Y1)
    Gx2, Gy2 = np.meshgrid(X2,Y2)
    positions1 = np.vstack([Gy1.ravel(), Gx1.ravel()])
    positions2 = np.vstack([Gy2.ravel(), Gx2.ravel()])
    positions = np.hstack([positions1,positions2])
    return positions



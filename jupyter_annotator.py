import numpy as np
from bokeh.models import PolyDrawTool,PolyEditTool,FreehandDrawTool
from bokeh.plotting import figure, show

def read_hne(
    path
):
    """
        Read H&E image 
        
        Parameters
        ----------     
        path 
            path to image, must follow supported Pillow formats - https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
        categorical_covariate_keys

    """
    from PIL import Image
    im = Image.open(path)
    im = im.convert("RGBA")
    return np.array(im)


def scribbler(
    imarray,
    anno_order,
    anno_colors,
):
    """
        interactive scribble line annotations with Bokeh  
        
        Parameters
        ----------     
        imarray  
            image in numpy array format (nparray)
        anno_order
            list of structures to annotate (list of strings)
        anno_colors
            list of colors for the structures in anno_order, must have the same length (list of strings)

    """

    
    imarray_c = imarray[:,:].copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])

    p =  figure(width=1000,height=1000)
    plotted_image = p.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    anno_color_map = dict(zip(anno_order, anno_colors))
    anno_color_map
    render_dict = {}
    draw_tool_dict = {}
    for l in anno_order:
        render_dict[l] = p.multi_line([], [], line_width=5, alpha=0.4, color=anno_color_map[l])
        draw_tool_dict[l] = FreehandDrawTool(renderers=[render_dict[l]], num_objects=50)
        draw_tool_dict[l].description = l
        p.add_tools(draw_tool_dict[l])
    
    
    return p, render_dict
    
def complete_pixel_gaps(x,y):
    from scipy import interpolate
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

        training_labels[np.array(xs).astype(int),np.array(ys).astype(int)] = idx+1
        # df = pd.DataFrame(render_dict[a].data_source.data)
        # df.index = a+'-'+df.index.astype('str')
        # annotations = pd.concat([annotations,df])
    training_labels = training_labels.transpose()
    import skimage as sk 
    return sk.segmentation.expand_labels(training_labels, distance=10)


def rgb_from_labels(labelimage,colors):

    labelimage_rgb = np.zeros((labelimage.shape[0],labelimage.shape[1] ,4))
    from PIL import ImageColor
    for c in range(len(colors)):
        color = ImageColor.getcolor(colors[c], "RGB")
        labelimage_rgb[np.where(labelimage == c+1)[0],np.where(labelimage == c+1)[1],0:3] = np.array(color)
    labelimage_rgb[:,:,3] = 255
    return labelimage_rgb.astype('uint8')


def sk_rf_classifier(
    im,
    training_labels
    
):
    from skimage import data, segmentation, feature, future
    from sklearn.ensemble import RandomForestClassifier
    from functools import partial

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
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.rcParams["figure.dpi"] = 100
    out_img = np.zeros(im1.shape,dtype=im1.dtype)
    out_img[:,:,:] = (alpha * im1[:,:,:]) + ((1-alpha) * im2[:,:,:])
    out_img[:,:,3] = 255
    if show:
        plt.imshow(out_img)
    return out_img


    
def annotator(
    imarray,
    annotation,
    anno_order,
    anno_colors,
    fig_downsize_factor = 10,
    
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
        anno_order
            list of structures to annotate (list of strings)
        anno_colors
            list of colors for the structures in anno_order, must have the same length (list of strings)
        fig_downsize_factor
            a plotting thing

    """
    
    from bokeh.models import PolyDrawTool,PolyEditTool,FreehandDrawTool
    from bokeh.plotting import figure, show
    from bokeh.models import TabPanel, Tabs

    # tab1
    imarray_c = annotation[:,:].copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])
    p = figure(width=int(imarray_c.shape[1]/fig_downsize_factor),height=int(imarray_c.shape[0]/fig_downsize_factor))
    plotted_image = p.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    tab1 = TabPanel(child=p, title="Annotation")

    # tab2
    imarray_c = imarray[:,:].copy()
    np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])
    p1 = figure(width=int(imarray_c.shape[1]/fig_downsize_factor),height=int(imarray_c.shape[0]/fig_downsize_factor), x_range=p.x_range,y_range=p.y_range)
    plotted_image = p1.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    tab2 = TabPanel(child=p1, title="Image")

    # # tab3
    # imarray_c = result_rgb[:,:].copy()
    # np_img2d = imarray_c.view("uint32").reshape(imarray_c.shape[:2])
    # p2 = figure(width=int(imarray_c.shape[1]/fig_downsize_factor),height=int(imarray_c.shape[0]/fig_downsize_factor), x_range=p.x_range,y_range=p.y_range)
    # plotted_image = p2.image_rgba(image=[np_img2d], x=0, y=0, dw=imarray_c.shape[1], dh=imarray_c.shape[0])
    # tab3 = TabPanel(child=p2, title="Annotation")
    anno_color_map = dict(zip(anno_order, anno_colors))
    anno_color_map

    # brushes
    render_dict = {}
    draw_tool_dict = {}
    for l in anno_order:
        render_dict[l] = p.multi_line([], [], line_width=5, alpha=0.4, color=anno_color_map[l])
        draw_tool_dict[l] = FreehandDrawTool(renderers=[render_dict[l]], num_objects=50)
        draw_tool_dict[l].description = l
        p.add_tools(draw_tool_dict[l])

    tabs = Tabs(tabs=[tab1, tab2])
    return tabs, render_dict


def update_annotator(
    imarray,
    result,
    anno_colors,
    render_dict,
):
    """
        updates annotations and generates overly (out_img) and the label image (corrected_labels)
        
        Parameters
        ----------     
        imarray  
            image in numpy array format (nparray)
        result  
            label image in numpy array format (nparray)
        anno_colors
            list of colors for the structures in anno_order, must have the same length (list of strings)
        render_dict
            bokeh data container

    """
    
    from skimage.draw import polygon
    corrected_labels = result.copy()
    # annotations = pd.DataFrame()
    for idx,a in enumerate(render_dict.keys()):
        if render_dict[a].data_source.data['xs']:
            print(a)
            for o in range(len(render_dict[a].data_source.data['xs'])):
                x = np.array(render_dict[a].data_source.data['xs'][o]).astype(int)
                y = np.array(render_dict[a].data_source.data['ys'][o]).astype(int)
                rr, cc = polygon(y, x)
                corrected_labels[rr, cc] = idx+1

    # corrected_labels = corrected_labels.transpose()     
    #generate overlay image
    rgb = rgb_from_labels(corrected_labels,anno_colors)
    out_img = overlay_lebels(imarray,rgb,alpha=0.8,show=False)
    # out_img = out_img.transpose() 
    return out_img, corrected_labels

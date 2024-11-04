"""
Functions to plot the stack results.

"""
from typing import Union, Optional, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Ellipse, Arc

def arg_list(arg: Any, n: int=1) -> List[Any]:
    """
    convert an arg to a list with length n.
    if arg is not a list, return a list with arg repeated n times.
    """
    return [arg]*n if not isinstance(arg, list) else arg

def get_ticks_labels(
        shape: int, 
        lim: Union[Tuple[float, float], List[float]], 
        step: int = 20, 
        digits: int = 1
        ) -> Tuple[List[int], List[str]]:
    """
    get the ticks and tick labels for a range.
    
    Args:
        shape (int): the number of ticks.
        lim (Tuple[float, float]): the range of ticks.
        step (int, optional): the step of ticks. Defaults to 20.
        digits (int, optional): the number of digits of tick labels. Defaults to 1.
    Returns:
        Tuple(List[int], List[str]): the ticks and tick labels.
    """
    ticks = range(0, shape +1, step)
    tick_labels = np.linspace(lim[0], lim[1], len(ticks))
    tick_labels = [f'%.{digits}f'%tick for tick in tick_labels]
    return list(ticks), tick_labels

def set_ticks(
        ax: Axes, 
        xticks: Optional[List[int]] = None, 
        xticklabels: Optional[List[str]] = None,
        yticks: Optional[List[int]] = None, 
        yticklabels: Optional[List[str]] = None,
        share: bool = True, 
        **kargs
        ) -> Axes:
    """
    set the ticks and ticklabels for x and y axis.
    
    If share is True, x and y axis are shared the same ticks.
    """
    if xticks is None and yticks is None:
        raise ValueError('xticks and yticks cannot be both None')
    
    if xticks is not None:
        ax.set_xticks(xticks, xticklabels, **kargs)
        if share:
            ax.set_yticks(xticks, xticklabels, **kargs)
    if yticks is not None:
        ax.set_yticks(yticks, yticklabels, **kargs)
        if share:
            ax.set_xticks(yticks, yticklabels, **kargs)

    return ax

def get_colorbar_cax(
        fig: Figure, 
        ax: Axes, 
        loc: str = 'right', 
        w_pad: float = 0.01, 
        w_factor: float = 0.05, 
        h_pad:float = 0.01, 
        h_factor:float = 0.05,
        **kwargs
        ) -> Axes:
    """
    get colorbar axes.
    """
    # Get the bounding box of the axes
    bbox = ax.get_position()
    
    # Calculate the position of the colorbar
    if loc == 'right':
        cax_x0 = bbox.x1 + w_pad
        cax_y0 = bbox.y0
        cax_width = bbox.width * w_factor
        cax_height = bbox.height
    elif loc == 'top':
        cax_x0 = bbox.x0
        cax_y0 = bbox.y1 + h_pad
        cax_width = bbox.width
        cax_height = bbox.height * h_factor
        
    else:
        raise ValueError(f"colorbar location = {loc} is supported")
    
    # Create the colorbar axes
    return fig.add_axes((cax_x0, cax_y0, cax_width, cax_height), **kwargs)


def plot_heatmap(
        data: np.ndarray, 
        fig: Optional[Figure]=None, 
        ax: Optional[Axes]=None, 
        cmap: str='jet', 
        cbt: Optional[str]=None, 
        norm = None, 
        title: Optional[str]=None, 
        xlabel: Optional[str]=None, 
        ylabel: Optional[str]=None
        ):
    """
    plot heatmap with colorbar.
    
    Args:
        data: 2D data to plot.
        fig (matplotlib.figure.Figure, optional): figure to plot. Defaults is None, create a new figure.
        ax (matplotlib.axes.Axes, optional): axis to plot. Defaults is None, create a new axis.
        cmap (str, optional): color map. Defaults is 'jet'.
        cbt (str, optional): colorbar title. Defaults is None.
        norm (str, optional): colorbar norm. Defaults is None, use 'linear'. \
            
            str type is **NOT** supported in low python verision(e.g. 3.8).\
                Use normalize from matplolib.colors instead.
        title (str, optional): plot title. Defaults is None.
        xlabel (str, optional): x axis label. Defaults is None.
        ylabel (str, optional): y axis label. Defaults is None.
    Returns:
        figure, axes, pcolormesh object, and colorbar object if cbt is not None
 
    """
    if fig is None:
        fig, ax = plt.subplots()
    
    pcm = ax.pcolormesh(data, cmap=cmap, norm=norm) # type: ignore
    ax.set_aspect('equal', 'box') # type: ignore
    
    if title is not None:
        ax.set_title(title) # type: ignore
    if xlabel is not None:
        ax.set_xlabel(xlabel) # type: ignore
    if ylabel is not None:
        ax.set_ylabel(ylabel) # type: ignore
    
    if cbt is not None:
        cax = get_colorbar_cax(fig, ax, loc='right') # type: ignore
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(pcm, cax=cax, label=cbt)
        return fig, ax, pcm, cbar
    else:
        return fig, ax, pcm


def plot_heatmaps(
        data: List[np.ndarray], 
        fig: Optional[Figure] = None, 
        axes: Union[List[Axes], Tuple[Axes], None]=None, 
        cmap: Union[str, list]= 'jet', 
        cbt: Union[str, list, None]=None, 
        norm =None, 
        sharey: bool=True, 
        title: Union[str, list, None]=None,
        xlabel: Union[str, list]='X', 
        ylabel: Union[str, list]='Y'
        ):
    """
    plot multiple heatmaps in a row.
    
    Notes:
        All args (except fig and axes) are list, and the length of the list is the number of heatmaps to plot.
        If an arg is not a list, it will be converted to a list with arg be repeated.
    
    Args:
        data: list of 2d numpy array.
        sharey: bool, share y axis or not.
        norm: matplotlib.color.norm (or str, may not suitable for low python version), or a list of norm. 
        Default is None, use 'linear' for all maps. 
        Other parameters are similar to norm.
    Returns:
        fig, axes
    """
    n = len(data)
    cmap = arg_list(cmap, n)
    cbt = arg_list(cbt, n)
    norm = arg_list(norm, n)
    title = arg_list(title, n)
    xlabel = arg_list(xlabel, n)
    ylabel = arg_list(ylabel, n)
    
    if fig is None:
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=sharey)
    
    if sharey:
        axes[0].set_ylabel(ylabel[0])   # type: ignore
          
        for i ,ax in enumerate(list(axes)): # type: ignore
            plot_heatmap(data[i], fig=fig, ax=ax, cmap=cmap[i], cbt=cbt[i], norm=norm[i], 
                        title=title[i], xlabel=xlabel[i])
    else:
        for i ,ax in enumerate(list(axes)): # type: ignore
            plot_heatmap(data[i], fig=fig, ax=ax, cmap=cmap[i], cbt=cbt[i], norm=norm[i], 
                        title=title[i], xlabel=xlabel[i], ylabel=ylabel[i])
    return fig, axes
    
def plot_stack_fit_residual(
        data:list, 
        fig=None, 
        axes:Union[list, tuple, None]=None, 
        cmap:Union[str, list]='jet', cbt:Union[str, list]=r'T[$\mu$K]', 
        norm:Union[str, list]=['log','log','linear'], sharey=True, 
        title:Union[str, list]=['Pairwise-stacked map','Fitted halo contribution','Filament signal'],
        xlabel:Union[str, list]='X', 
        ylabel:Union[str, list]='Y',
        xticks:Optional[list]=None, 
        xticklabels:Optional[list]=None, 
        fit_mask:Optional[np.ndarray]=None, alpha=0.15
        ):
    
    """
    plot stack, fit and residual map.
    """
    
    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=sharey)
        
    fig, axes= plot_heatmaps(data, fig=fig, axes=axes, cmap=cmap, cbt=cbt, norm=norm, 
                             sharey=sharey, title=title, xlabel=xlabel, ylabel=ylabel)
    
    if fit_mask is not None:
        axes[1].pcolormesh(fit_mask, cmap='gray', alpha=alpha)  # type: ignore ## Plot mask region
    
    if xticks is not None:
        set_ticks(axes[0], xticks=xticks, xticklabels=xticklabels) # type: ignore
        set_ticks(axes[1], xticks=xticks, xticklabels=xticklabels) # type: ignore
        set_ticks(axes[2], xticks=xticks, xticklabels=xticklabels) # type: ignore
    
    return fig, axes

def plot_residuals(
        data:list,
        fig=None,
        axes = None,
        sharey=True,
        norm=Normalize(-20, 60), 
        w_pad=0,
        cmap='jet',
        cbt=r'T[$\mu$K]',
        title=['HI only','HI + noise'],
        xlabel='X',
        ylabel='Y'
        ):

    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=sharey)

    for i, ax in enumerate(list(axes)): # type: ignore
        _,_,pcm,*_ = plot_heatmap(data[i], fig=fig, ax=ax, cmap=cmap, norm=norm, title=title[i], xlabel=xlabel, ylabel=ylabel) # type: ignore
        ax.label_outer()
        
    fig.tight_layout(w_pad=w_pad)
    
    cax = get_colorbar_cax(fig, ax)
    cbar = plt.colorbar(pcm, cax=cax, label=cbt)
        
    return fig, axes, cbar

def plot_line(
        x:list,
        y:list, 
        fig=None,
        ax = None, 
        xlabel:Optional[str]=None, 
        ylabel:Optional[str]=None, 
        title:Optional[str]=None, 
        color:Union[str, list, None]=None, 
        marker:Union[str, list, None]=None, 
        linestyle:Union[str, list, None]=None, 
        label:Union[str, list, None]=None
        ):
    """
    plot a line or lines.
    
    Args:
        x (list): x data points.
        y (list): y data points.
        fig (matplotlib.figure.Figure, optional): figure object. Defaults is None, create a new figure.
        ax (matplotlib.axes.Axes, optional): axes object. Defaults is None, use the axes of new created figure.
        xlabel (str, optional): x label. Defaults is None.
        ylabel (str, optional): y label. Defaults is None.
        title (str, optional): title. Defaults is None.
        color (str, list, optional): color of the line. Defaults is None.
        marker (str, list, optional): marker of the line. Defaults is None.
        linestyle (str, list, optional): linestyle of the line. Defaults is None.
        label (str, list, optional): label of the line. Defaults is None.
    Returns:
        (fig and axes): figure and axes object.
    """
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    x_array = np.array(x)
    
    if len(x_array.shape) == 1:
        x_array = x_array.reshape(1, -1)
        
    n, m = x_array.shape
    
    color = arg_list(color, n)
    marker = arg_list(marker, n)
    linestyle = arg_list(linestyle, n)
    label = arg_list(label, n)
    for i in range(n):
        ax.plot(x_array[i], y[i], color=color[i], linestyle=linestyle[i], marker=marker[i], label=label[i]) # type: ignore
    
    if xlabel is not None:
        ax.set_xlabel(xlabel) # type: ignore
    if ylabel is not None:
        ax.set_ylabel(ylabel) # type: ignore
    if title is not None:
        ax.set_title(title) # type: ignore
    if label is not None:
        ax.legend() # type: ignore
        
    return fig, ax

def plot_line_2c(
        x:list, 
        y:list, 
        cut:int=1, 
        fig=None, 
        axes = None, 
        sharey=True,
        xlabel:Union[str, list, None]=None, 
        ylabel:Union[str, list, None]=None,
        title:Union[str, list, None]=None, 
        color:Union[str, list, None]=None,
        marker:Union[str, list, None]=None, 
        linestyle:Union[str, list, None]=None, 
        label:Union[str, list, None]=None
        ):
    """
    plot profiles in two column subplots.
    
    Parameters:
        x (list): x data points.
        y (list): y data points.
        cut (int): separate the data points into two parts, 
        e.g. cut=2 means the data points in the first two rows are plotted in the first subplot, 
        and rest are plotted in the second subplot.
    """
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 4), sharey=sharey)
    
    try:
        x_left, x_right = x[:cut], x[cut:]
        y_left, y_right = y[:cut], y[cut:]
    except IndexError as ie:
        raise ValueError('cut index is out of the range.') from ie
    
    n = len(x)
    xlabel = arg_list(xlabel, 2)
    ylabel = arg_list(ylabel, 2)
    title = arg_list(title, 2)
    
    color = arg_list(color, n)
    marker = arg_list(marker, n)
    linestyle = arg_list(linestyle, n)
    label = arg_list(label, n)
    
    for i, ax in enumerate(list(axes)): # type: ignore
        c = color[:cut] if i==0 else color[cut:]
        m = marker[:cut] if i==0 else marker[cut:]
        l = linestyle[:cut] if i==0 else linestyle[cut:]
        lb = label[:cut] if i==0 else label[cut:]
        
        plot_line(x_left if i==0 else x_right, y_left if i==0 else y_right, 
                    fig=fig, ax=ax, xlabel=xlabel[i], ylabel=ylabel[i],
                    title=title[i], color=c, marker=m, linestyle=l, label=lb)
        
    if sharey:
        fig.subplots_adjust(wspace=0)
        
    return fig, axes


def plot_lines_2c2r(
        x:List[list],
        y:List[list],
        cut:List=[2,1],
        fig=None,
        axes= None,
        sharex:bool=True,
        sharey:bool=True, 
        xlabel:Optional[List[list]]=None,
        ylabel:Optional[List[list]]=None, 
        title:Optional[List[list]]=None,
        color:Optional[list]=None, 
        marker:Optional[list]=None,
        linestyle:Optional[list]=None, 
        label:Optional[list]=None
        ):
    """ 
    plot lines in two column two row subplots. The top and bottom rows are shared the same args.
    
    Args:
        x (List[list]): x data points.
        y (List[list]): y data points.
        cut (List[int]): separate the data points into two parts, \
            e.g. cut=[2,1] means the data points in the first two row \
                are plotted in the top two subplots respectively, \
                    and rest are plotted in the bottom two subplots, \
                        with one in left panel,and the others in the right panel.
        sharex (bool): share x-axis between subplots.
        sharey (bool): share y-axis between subplots.
        xlabel (List[list]): x-axis labels.
        ylabel (List[list]): y-axis labels.
        title (List[list]): subplot titles.
        color (list): line colors.
        marker (list): line markers.
        linestyle (list): line styles.
        label (list): line labels.
    Returns:
        (fig, axes): figure and axes objects.
    """

    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=sharex, sharey=sharey)

    try:
        x_top, x_bottom = x[:cut[0]], x[cut[0]:]
        y_top, y_bottom = y[:cut[0]], y[cut[0]:]
    except IndexError as ie:
        raise ValueError('first cut index is out of the range.') from ie

    n = len(x)
    xlabel_list = arg_list(xlabel, 2)
    ylabel_list = arg_list(ylabel, 2)
    title_list = arg_list(title, 2)
    
    color_list = arg_list(color, n)
    marker_list = arg_list(marker, n)
    linestyle_list = arg_list(linestyle, n)
    label_list = arg_list(label, n)
    
    for i in range(2):
        ax = list(axes)[i] # type: ignore
        x_data = x_top if i==0 else x_bottom
        y_data = y_top if i==0 else y_bottom
        c = color_list[:cut[0]] if i==0 else color_list[cut[0]:]
        m = marker_list[:cut[0]] if i==0 else marker_list[cut[0]:]
        l = linestyle_list[:cut[0]] if i==0 else linestyle_list[cut[0]:]
        lb = label_list[:cut[0]] if i==0 else label_list[cut[0]:]
        plot_line_2c(x_data, y_data, cut=cut[1], sharey=sharey,
                    fig=fig, axes=ax, xlabel=xlabel_list[i], ylabel=ylabel_list[i],
                    title=title_list[i], color=c, marker=m, linestyle=l, label=lb)
    if sharex:
        fig.subplots_adjust(hspace=0)
    if sharey:
        fig.subplots_adjust(wspace=0)
        
    return fig, axes

def plot_profiles(
        x:list, y:list, 
        cut:list=[4,2],
        fig=None,
        axes=None,
        sharey=True,
        sharex=True,
        xlabel:List[list]=[[None]*2, ['Y','X']], 
        ylabel:list=[[r'$T$[$\mu$K] (HI only)',None],[r'$T$[$\mu$K] (HI + noise)',None]], 
        title:List[list]=[['Transverse-section profile','Lengthwise-section profile'],[None]*2], 
        color:List=['b','r','r','k']*2,
        marker:list=['.','','o','o']*2,
        linestyle:list=['-','--','','']*2, 
        label:List=[r'$|{\rm X}|< 0.5$', 'Gaussian fit', r'$T_{\rm f}$',r'$T_{\rm bg}$']+[None]*4,
        text_pos:List[list]=[[-0.4, 30],[-0.4, 10],[-2, 30]], 
        width:list=[0.32]*2, 
        fontsize='14'
        ):
    """
    plot profiles in two column two row subplots. \
        The top and bottom rows are shared the same args.

    Args:
        x (List[list]): x-axis data.
        y (List[list]): y-axis data.
        cut (List[int]): cut index for x and y data.
        fig (Figure, optional): figure object. Defaults to None.
        axes (Axes, optional): axes object. Defaults to None.
        sharey (bool, optional): share y-axis between subplots. Defaults to True.
        sharex (bool, optional): share x-axis between subplots. Defaults to True.
        xlabel (List[list]): x-axis labels.
        ylabel (List[list]): y-axis labels.
        title (List[list]): subplot titles.
        color (List[str]): line colors.
        marker (List[str]): line markers.
        linestyle (List[str]): line styles.
        label (List[str]): line labels.
        text_pos (List[list]): position of text.
        width (List[float]): width of filament.
        fontsize (str, optional): fontsize of text. Defaults to '14'.
    Returns:
        (fig, axes): figure and axes objects.
    """
    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(9,6), sharex=sharex, sharey=sharey)
    
    plot_lines_2c2r(x, y, cut=cut, fig=fig, axes=axes, sharex=sharex, sharey=sharey,
                     xlabel=xlabel, ylabel=ylabel, title=title, color=color, 
                     marker=marker, linestyle=linestyle, label=label)
    
    # addtional settings
    for i, ax in enumerate(axes.flatten()): # type: ignore
        # add horizontal lines that marker T = 0
        ax.axhline(0, linestyle='--', c='gray') 
        
        # add text that marker the mean values of filament and background.
        if (i==1) or (i > 2):
            # filament
            mf, sf = np.mean(y[2*i]), np.std(y[2*i])
            strf = r'$<T_{\rm f}>$ = %.1f $\mu$K, $\sigma_{T_{\rm f}}$ = %.1f $\mu$K'%(mf,sf)
            ax.text(*text_pos[0], strf, color='r', fontsize=fontsize) 
            # background
            mb, sb = np.mean(y[2*i+1]), np.std(y[2*i+1])
            strb = r'$<T_{\rm bg}>$ = %.1f $\mu$K, $\sigma_{T_{\rm bg}}$ = %.1f $\mu$K'%(mb,sb)
            ax.text(*text_pos[1], strb, color='k', fontsize=fontsize)
        # add text that marker the width of filament
        else:
            ax.text(*text_pos[2], r'$w_{\rm t} = %.2f$'%width[i//2], color='r', fontsize=fontsize)
    
    return fig, axes


def plot_hist(
        data:list, 
        bins:Optional[list]=None, 
        fig=None, ax=None, 
        label:Union[str, list, None]=None, 
        color:Union[str, list, None]=None, 
        density:Union[bool, list]=True, 
        histtype:Union[str, list]='step', 
        title:Optional[str]=None, 
        xlabel:Optional[str]=None, 
        ylabel='PDF',
        **kwargs
        ):
    """
    plot the histogram, defualt is step density histogram
    """
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(6,4))
    
    data_array = np.array(data)
    if len(data_array.shape) == 1:
        data_array = data_array.reshape(-1,1)
    n,_ = data_array.shape
    
    bins = arg_list(bins, n)
    label = arg_list(label, n)
    color = arg_list(color, n)
    density = arg_list(density, n)
    histtype = arg_list(histtype, n)
    
    for i in range(n):
        ax.hist(data_array[i], bins=bins[i], label=label[i], color=color[i], density=density[i], histtype=histtype[i], **kwargs) # type: ignore
    
    if title is not None:
        ax.set_title(title) # type: ignore
    if xlabel is not None:
        ax.set_xlabel(xlabel) # type: ignore
    if ylabel is not None:
        ax.set_ylabel(ylabel) # type: ignore
        
    if label is not None:
        ax.legend() # type: ignore
    
    return fig, ax

def plot_hist_2c(
        data:list, 
        bins=None, 
        cut:int=1, 
        fig=None, 
        axes=None, 
        sharey=True,
        label=None, 
        color=None, 
        density=True, 
        histtype='step', 
        title=None, xlabel=None, 
        ylabel=['PDF',None], 
        **kwargs
        ):
    """
    plot the histogram in two column subplots, defualt is step density histogram
    """
    
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=sharey)

    try:
        data_left, data_right = data[:cut], data[cut:]
    except IndexError as ie:
        raise ValueError('cut index is out of the range.') from ie
    
    n = len(data)
    xlabel = arg_list(xlabel, 2)
    ylabel = arg_list(ylabel, 2)
    title = arg_list(title, 2)
    bins = arg_list(bins, n)
    label = arg_list(label, n)
    color = arg_list(color, n)
    density = arg_list(density, n)
    histtype = arg_list(histtype, n)
    
    
    for i, ax in enumerate(list(axes)): # type: ignore
        b = bins[:cut] if i==0 else bins[cut:]
        c = color[:cut] if i==0 else color[cut:]
        d = density[:cut] if i==0 else density[cut:]
        ht = histtype[:cut] if i==0 else histtype[cut:]
        lb = label[:cut] if i==0 else label[cut:]
        
        plot_hist(data_left if i==0 else data_right, bins=b, fig=fig, ax=ax, 
                  label=lb, color=c, density=d, histtype=ht, title=title[i],
                  xlabel=xlabel[i], ylabel=ylabel[i], **kwargs)
        
    if sharey:
        fig.subplots_adjust(wspace=0)
        
    return fig, axes

def plot_hist_result(
        data:list, 
        bins:Optional[list]=None, 
        cut:int=4, 
        fig=None, 
        axes=None, 
        sharey=True, 
        label:list=[None]*4+['C','L','R','B'], 
        color:list=['r','b','g','k']*2, 
        density=True, 
        histtype='step', 
        title:list=['HI only','HI + noise'], 
        xlabel=r'$T$[$\mu$K]', 
        ylabel=['PDF',None],
        **kwargs
        ):
    """
    plot the histogram results in two column subplots, defualt is step density histogram.
    """
    
    fig, axes = plot_hist_2c(data, bins=bins, cut=cut, sharey=sharey, 
                             fig=fig, axes=axes, label=label, color=color, 
                             density=density, histtype=histtype, title=title, 
                             xlabel=xlabel, ylabel=ylabel, **kwargs)
    
    for ax in axes: # type: ignore
        ax.axvline(0, linestyle='--', c='gray', zorder=-1000) 
        
    return fig, axes
    
def plot_ellipses(
        ax,
        xy_pos:list[Tuple[float,float]]=[(0,0)],
        width: Union[float, list[float]]=1,
        height:Union[float, list[float], None]=None,
        angle: Union[float, list[float]]=0,
        **kwargs
        ):
    """
    Plot ellipses (or circles) on the given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to plot the ellipses.
        xy_pos (list[tuple(float, float)]): xy positions.
        width  (float, list): widths of the ellipses.
        height (float, list): heights of the ellipses, default is None, use the width.
        angle  (float, list): rotation of the ellipse in degrees (**counterclockwise**)
        theta1,theta2 (float, list): starting and ending angles in degrees.
        **kwargs: additional keyword arguments.
    """
    if height is None:
        height = width
    
    # find the number of ellipse need to plot
    n = len(xy_pos)
    if n == 1:
        e = Ellipse(xy_pos[0], width, height, angle=angle, **kwargs) # type: ignore
        ax.add_patch(e)
    else:
        width_list  = arg_list(width,  n)
        height_list = arg_list(height, n)
        angle_list  = arg_list(angle,  n)

        for xy,w,h,a in zip(xy_pos, width_list, height_list, angle_list):
            e = Ellipse(xy, w, h, angle=a,**kwargs)
            ax.add_patch(e)

def plot_arcs(
        ax,
        xy_pos: list[Tuple[float,float]]=[(0,0)],
        width:  Union[float, list[float]]=1,
        height: Union[float, list[float], None]=None,
        angle:  Union[float, list[float]]=0,
        theta1: Union[float, list[float]]=0,
        theta2: Union[float, list[float]]=360,
        **kwargs
        ):
    """
    Plot arcs on the given axis. 

    Args:
        ax (matplotlib.axes.Axes): The axis on which to plot the ellipses.
        xy_pos (list[tuple(float, float)]): xy positions.
        width  (float, list): widths of the ellipses.
        height (float, list): heights of the ellipses, default is None, use the width.
        angle  (float, list): rotation of the ellipse in degrees (**counterclockwise**)
        theta1,theta2 (float, list): starting and ending angles in degrees.
        **kwargs: additional keyword arguments.
    """
    if height is None:
        height = width
    
    # find the number of ellipse need to plot
    n = len(xy_pos)
    if n == 1:
        a = Arc(xy_pos[0], width, height, angle=angle, theta1=theta1, theta2=theta2, **kwargs) # type: ignore
        ax.add_patch(a)
    else:
        width_list  = arg_list(width,  n)
        height_list = arg_list(height, n)
        angle_list  = arg_list(angle,  n)
        theta1_list = arg_list(theta1, n)
        theta2_list = arg_list(theta2, n)

        for xy,w,h,a,t1,t2 in zip(xy_pos, width_list, height_list, angle_list, theta1_list, theta2_list):
            a = Arc(xy, w, h, angle=a, theta1=t1, theta2=t2, **kwargs)
            ax.add_patch(a)
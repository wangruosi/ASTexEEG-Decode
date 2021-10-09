
def set_rcParams(mode, update_kwargs=None):
    import matplotlib as mpl
    if mode == 'paper':
        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams['figure.dpi'] = 500
        mpl.rcParams['savefig.dpi'] = 500
        mpl.rcParams['font.size'] = 7
        mpl.rcParams['ytick.labelsize'] = 6
        mpl.rcParams['xtick.labelsize'] = 6
        mpl.rcParams['lines.linewidth'] = .8
        mpl.rcParams['lines.markersize'] = 2
    
    elif mode == 'notebook':
        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams['font.size'] = 7
        mpl.rcParams['ytick.labelsize'] = 6
        mpl.rcParams['figure.dpi'] = 150
        mpl.rcParams['xtick.labelsize'] = 6
        mpl.rcParams['lines.linewidth'] = .8
        mpl.rcParams['lines.markersize'] = 2
    
    if update_kwargs is not None:
        mpl.rcParams.update(update_kwargs)
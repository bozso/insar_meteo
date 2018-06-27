from inmet.gnuplot import Gnuplot

def plot_scatter(data, ncols, out, title="", idx=None, titles=None,
                 palette="rgb", endian="default", portrait=False):
    
    binary = "%double" * (ncols + 2)
    
    if idx is None:
        idx = range(ncols)
    
    # parse titles
    if titles is None:
        titles = range(1, ncols + 1)
    
    gpt = Gnuplot(out=out, term="pngcairo font ',8' size 800,600", debug=True)
    gpt.margins(rmargin=0.1, screen=True)
    gpt.multiplot(len(idx), title=title, portrait=portrait)
    # gpt.binary("format='{}' endian={}".format(binary, endian))
    
    gpt.xtics(0.4)
    
    # gpt.palette(palette)
    # gpt.colorbar(cbfomat="%4.2f")
    
    gpt.unset("colorbox")
    
    for ii in idx:
        inp_format = "1:2:{}".format(ii + 3)
        gpt.plot(data, using=inp_format, palette=True, binary=binary,
                 endian=endian, pt_type="dot")
        gpt.end_plot()

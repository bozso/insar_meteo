from inmet.gmt import GMT, _gmt_five, get_version, gmt
import inmet.cwrap as cw

def plot_scatter(infile, ncols, out=None, xy_range=None, z_range=None,
                 idx=None, titles=None, cpt="drywet", x_axis="a0.5g0.25f0.25",
                 y_axis="a0.25g0.25f0.25", mode="v", offset=25.0, step=None,
                 proj="M", tryaxis=False, **kwargs):

    # default is "${infile}.png"
    if out is None:
        out = pth.basename(infile).split(".")[0] + ".png"
    
    name, ext = pth.splitext(out)
    
    if ext != ".ps":
        ps_file = name + ".ps"
    else:
        ps_file = out
    
    # 2 additional coloumns for coordinates, float64s are expected
    bindef = "{}d".format(ncols + 2)
    
    if xy_range is None or z_range is None:
        _xy_range, _z_range = get_ranges(data=infile, binary=bindef)
    
    if xy_range is None:
        xy_range = _xy_range

    if z_range is None:
        z_range = _z_range

    if idx is None:
        idx = range(args.ncols)
    
    # parse titles
    if titles is None:
        titles = range(1, args.ncols + 1)
    
    gmt = GMT(ps_file, R=xy_range)
    x, y = gmt.multiplot(len(idx), proj, nrows=kwargs.pop("nrows"), **kwargs)
    
    gmt.makecpt("tmp.cpt", C=cpt, Z=True, T=z_range)
    
    # do not plot the scatter points yet just test the placement of basemaps
    if tryaxis:
        for ii in idx:
            input_format = "0,1,{}".format(ii + 2)
            gmt.psbasemap(Xf="{}p".format(x[ii]), Yf="{}p".format(y[ii]),
                          B="WSen+t{}".format(titles[ii]), Bx=x_axis, By=y_axis)
    else:
        for ii in idx:
            input_format = "0,1,{}".format(ii + 2)
            gmt.psbasemap(Xf="{}p".format(x[ii]), Yf="{}p".format(y[ii]),
                          B="WSen+t{}".format(titles[ii]), Bx=x_axis, By=y_axis)

            gmt.psxy(data=infile, i=input_format, bi=bindef, S="c0.025c",
                     C="tmp.cpt")

    if step is None:
        label = "5"
    else:
        label = str(step)
    
    if label is not None:
        label += args.label
    
    gmt.colorbar(mode=mode, offset=offset, B=label, C="tmp.cpt")
    
    if ext != ".ps":
        gmt.raster(out, **kwargs)
        os.remove(ps_file)
    
    os.remove("tmp.cpt")
    
    del gmt

def hist(data, ps_file, binwidth=0.1, config=None, binary=None, 
         left=50, right=25, top=25, bottom=50, **flags):
    
    ranges = tuple(float(elem)
                   for elem in info(data, bi=binary, C=True).split())
    
    min_r, max_r = min(ranges), max(ranges)
    binwidth = (max_r - min_r) * binwidth
    
    width, height = get_paper_size(config.pop("PS_MEDIA", "A4"))
    
    proj="X{}p/{}p".format(width - left - right, height - top - bottom)
    
    gmt = GMT(ps_file, config=config, R=(min_r, max_r, 0.0, 100.0), J=proj)
    
    gmt.psbasemap(Bx="a{}".format(round(binwidth)), By=5,
                  Xf=str(left) + "p", Yf=str(bottom) + "p")
    
    gmt.pshistogram(data=data, W=binwidth, G="blue", bi=binary, Z=1)

    del gmt

def make_ncfile(self, header_path, ncfile, endian="small"):
    fmt     = cw.get_par("SAM_IN_FORMAT", header_path)
    dempath = cw.get_par("SAM_IN_DEM", header_path)
    nodata  = cw.get_par("SAM_IN_NODATA", header_path)
    
    rows, cols              = cw.get_par("SAM_IN_SIZE", header_path).split()
    delta_lat, delta_lon    = cw.get_par("SAM_IN_DELTA", header_path).split()
    origin_lat, origin_lon  = cw.get_par("SAM_IN_UL", header_path).split()
    
    xmin = float(origin_lon)
    xmax = xmin + float(cols) * float(delta_lon)

    ymax = float(origin_lat)
    ymin = ymax - float(rows) * float(delta_lat)
    
    lonlat_range = "{}/{}/{}/{}".format(xmin ,xmax, ymin, ymax)
    
    increments = "{}/{}".format(self.delta_lon, self.delta_lat)
    
    Cmd = "xyz2grd {infile} -ZTL{dtype} -R{ll_range} -I{inc} -r -G{nc}"\
          .format(infile=dempath, dtype=dem_dtypes[fmt], ll_range=lonlat_range,
                  inc=increments, nc=ncfile)
    
    if get_version() > _gmt_five:
        Cmd = "gmt " + Cmd
    
    cmd(Cmd)

def get_ranges(data, binary=None, xy_add=None, z_add=None):
    
    if binary is not None:
        info_str = gmt("info", data, ret=True, bi=binary, C=True).split()
    else:
        info_str = gmt("info", data, ret=True, C=True).split()
    
    ranges = tuple(float(data) for data in info_str)
    
    if xy_add is not None:
        X = (ranges[1] - ranges[0]) * xy_add
        Y = (ranges[3] - ranges[2]) * xy_add
        xy_range = (ranges[0] - xy_add, ranges[1] + xy_add,
                    ranges[2] - xy_add, ranges[3] + xy_add)
    else:
        xy_range = ranges[0:4]

    non_xy = ranges[4:]
    
    if z_add is not None:
        min_z, max_z = min(non_xy), max(non_xy)
        Z = (max_z - min_z) * z_add
        z_range = (min_z - z_add, max_z + z_add)
    else:
        z_range = (min(non_xy), max(non_xy))
        
    return xy_range, z_range

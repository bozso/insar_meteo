function out = gmt(fun, varargin)
    gmt_cmds = {'grdcontour', 'grdimage', 'grdvector', 'grdview', 'psbasemap', ...
    'psclip', 'pscoast', 'pscontour', 'pshistogram', 'psimage', 'pslegend', ...
    'psmask', 'psrose', 'psscale', 'pstext', 'pswiggle', 'psxy', 'psxyz', ...
    'gmtlogo', 'blockmean', 'blockmedian', 'blockmode', 'filter1d', ...
    'fitcircle', 'gmt2kml', 'gmt5syntax', 'gmtconnect', 'gmtconvert', ...
    'gmtdefaults', 'gmtget', 'gmtinfo', 'gmtlogo', 'gmtmath', 'gmtregress', ...
    'gmtselect', 'gmtset', 'gmtsimplify', 'gmtspatial', 'gmtvector', ...
    'gmtwhich', 'grd2cpt', 'grd2rgb', 'grd2xyz', 'grdblend', 'grdclip', ...
    'grdcontour', 'grdconvert', 'grdcut', 'grdedit', 'grdfft', 'grdfilter', ...
    'grdgradient', 'grdhisteq', 'grdimage', 'grdinfo', 'grdlandmask', ...
    'grdmask', 'grdmath', 'grdpaste', 'grdproject', 'grdraster', 'grdsample', ...
    'grdtrack', 'grdtrend', 'grdvector', 'grdview', 'grdvolume', 'greenspline', ...
    'kml2gmt', 'makecpt', 'mapproject', 'nearneighbor', 'project', 'sample1d', ...
    'spectrum1d', 'sph2grd', 'sphdistance', 'sphinterpolate', 'sphtriangulate', ...
    'splitxyz', 'surface', 'trend1d', 'trend2d', 'triangulate', 'xyz2grd'};
    
    split = strsplit(fun);
    
    if ismember(split{1}, gmt_cmds)
        out = gmt_cmd(fun, varargin{:});
    else
        switch (fun)
            case 'init'
                out = init(varargin{:});
            case 'finalize'
                finalize(varargin{:});
            case 'get_width'
                out = get_width(varargin{:});
            case 'get_height'
                out = get_height(varargin{:});
            case 'scale_pos'
                out = scale_pos(varargin{:});
            case 'colorbar'
                out = colorbar(varargin{:});
            otherwise
                error(['Unknown function ', fun]);
        end
    end
end % gmt

function out = init(psfile, varargin)
    validateattributes(psfile, {'char'}, {'nonempty'});
    
    args = struct('common', '', 'left', 10, 'right', 10, 'top', 10, 'bottom', 10, ...
                  'debug', 0);
    args = parseArgs(varargin, args, {'debug'});
    
    common  = args.common;
    left    = args.left;
    right   = args.right;
    top     = args.top;
    bottom  = args.bottom;
    
    validateattributes(common, {'char'}, {});

    klass = {'numeric'};
    attr = {'scalar', 'nonnan', 'positive', 'finite', 'real'};

    validateattributes(left, klass, attr);
    validateattributes(right, klass, attr);
    validateattributes(top, klass, attr);
    validateattributes(bottom, klass, attr);
    
    version = cmd('gmt --version');
    
    if strcmp(version(1), '5')
        is_five = true;
    elseif strcmp(version(1), '6')
        error('Cannot handle GMT 6.x!');
    elseif strcmp(version(1), '4')
        is_five = false;
    else
        error('GMT version is not 4,5 or 6!');
    end
    
    out.version = version;
    out.is_five = is_five;
    out.psfile = psfile;
    out.common = common;
    out.debug = args.debug;
    
    out.left = left;
    out.right = right;
    out.top = top;
    out.bottom = bottom;
    
    % sentinels
    out.commands = 'init';
    out.outfiles = 'init';
end % init

function finalize(Gmt)
    commands = strsplit(Gmt.commands, '\n');
    outfiles = strsplit(Gmt.outfiles, '\n');
    common = Gmt.common;
    
    if ~strcmp(commands{1}, 'init') | ~strcmp(outfiles{1}, 'init')
        error('Gmt struct was not initialized correctly!');
    end
    
    ncom = length(commands);
    
    idx = [];
    
    % get the indices of plotter functions
    for ii = 2:ncom
        if is_plotter(commands{ii})
            idx(end + 1) = ii;
        end
    end
    
    % if we have common flags add them
    if ~isempty(common)
        for ii = 2:ncom
            commands{ii} = [commands{ii}, ' ', common];
        end
    end
    
    % handle -O and -K options
    if numel(idx) > 1
        for ii = idx(1:end - 1)
            commands{ii} = [commands{ii}, ' -O'];
        end

        for ii = idx(2:end)
            commands{ii} = [commands{ii}, ' -K'];
        end
    end

    % append 'gmt ' to the command strings
    if Gmt.is_five
        for ii = 2:ncom
            commands{ii} = ['gmt ', commands{ii}];
        end
    end
    
    if Gmt.debug
        fprintf('DEBUG: GMT commands\n%s\n', strjoin(commands, '\n'));
    end
    
    return
    
    for ii = 2:ncom
        fid = sfopen(outfiles{ii}, 'a');
        
        out = cmd(commands{ii});
        
        fprintf(fid, '%s\n', out);
        fclose(fid);
    end
end % finalize

function Gmt = multiplot(Gmt, nplots, proj, varargin)
%
%             |       top           |    
%        -----+---------------------+----
%          l  |                     |  r
%          e  |                     |  i
%          f  |                     |  g
%          t  |                     |  h
%             |                     |  t
%        -----+---------------------+----
%             |       bottom        |    
    
    args = struct('xpad', 55, 'ypad', 75);
    args = parseArgs(varargin, args);
    
    xpad = args.xpad;
    ypad = args.ypad;
    
    klass = 'numeric';
    attr = {'scalar', 'positive', 'nonnan', 'real', 'finite'};
    validateattributes(xpad, klass, attr);
    validateattributes(ypad, klass, attr);
    
    nrows = ceil(sqrt(nplots) - 1);
    nrows = max(1, nrows);
    ncols = ceil(nplots / nrows),
    
    width = get_width(Gmt);

    % width available for plotting
    awidth = width - (left + right)
        
    % width of a single plot
    pwidth  = (awidth - (ncols - 1) * xpad) / ncols;
        
    Gmt.common = sprintf('%s -J%g%gp', Gmt.common, proj, pwidth);
        
    # height of a single plot
    pheight = get_height(Gmt)
        
    %# calculate psbasemap shifts in x and y directions
    %x = (left + ii * (pwidth + x_pad) for jj in range(nrows)
                                      %for ii in range(ncols))
    
    %y = (height - top - ii * (pheight + y_pad)
         %for ii in range(1, nrows + 1)
         %for jj in range(ncols))
    
    %# residual margin left at the bottom
    %self.bottom = height - top - nrows * pheight
    
    %return tuple(x), tuple(y)
end

function Gmt = gmt_cmd(Cmd, Gmt, outfile)
    Gmt.commands = sprintf('%s\n%s', Gmt.commands, Cmd);
    
    % outfile not defined
    if nargin == 2
        outfile = Gmt.psfile;
    end
    Gmt.outfiles = sprintf('%s\n%s', Gmt.outfiles, outfile);
end

function width = get_width(Gmt)
    
    if ismac
        null = '/dev/null';
    elseif isunix
        null = '/dev/null';
    elseif ispc
        null = 'NUL';
    else
        error('Could not identify platform!');
    end
    
    version = cmd('gmt --version');
    
    if Gmt.is_five
        Cmd = sprintf('gmt mapproject %s -Dp', Gmt.common);
    else
        Cmd = sprintf('mapproject %s -Dp', Gmt.common);
    end
    
    % before version 5.2
    if version(3) <= '1'
        Cmd = sprintf('%s %s -V', Cmd, null);
        out = strsplit(cmd(Cmd), '\n');
        
        for ii = 1:length(out)
            if ~isempty(strfind(out{ii}, 'Transform'))
                out = out{ii};
                break;
            end
        end
        
        if iscell(out)
            error('Keyword ''Transform'' not found in command output!');
        end
        
        outsplit = strsplit(out, '/');
        width = str2num(outsplit(5));
    else
        Cmd = sprintf('%s -Ww', Cmd);
        width = str2num(cmd(Cmd));
    end
end

function height = get_height(Gmt)
    
    if ismac
        null = '/dev/null';
    elseif isunix
        null = '/dev/null';
    elseif ispc
        null = 'NUL';
    else
        error('Could not identify platform!');
    end
    
    version = cmd('gmt --version');
    
    if Gmt.is_five
        Cmd = sprintf('gmt mapproject %s -Dp', Gmt.common);
    else
        Cmd = sprintf('mapproject %s -Dp', Gmt.common);
    end
    
    % before version 5.2
    if version(3) <= '1'
        Cmd = sprintf('%s %s -V', Cmd, null);
        out = strsplit(cmd(Cmd), '\n');
        
        for ii = 1:length(out)
            if ~isempty(strfind(out{ii}, 'Transform'))
                out = out{ii};
                break;
            end
        end
        
        if iscell(out)
            error('Keyword ''Transform'' not found in command output!');
        end
        
        outsplit = strsplit(out, '/');
        outsplit = strsplit(outsplit(6), ' ');
        height = str2num(outsplit(1));
    else
        Cmd = sprintf('%s -Wh', Cmd);
        height = str2num(cmd(Cmd));
    end
end

function out = scale_pos(Gmt, mode, varargin)
    
    args = struct('offset', 100, 'flong', 0.8, 'fshort', 0.2);
    args = parseArgs(varargin, args);
    
    left    = Gmt.left;
    right   = Gmt.right;
    top     = Gmt.top;
    bottom  = Gmt.bottom;

    width   = get_width(Gmt);
    height  = get_height(Gmt);
    
    offset  = args.offset;
    flong   = args.flong;
    fshort  = args.fshort;

    klass = 'numeric';
    attr = {'scalar', 'positive', 'nonnan', 'real', 'finite'};
    
    validateattributes(offset, klass, attr);
    validateattributes(flong, klass, attr);
    validateattributes(fshort, klass, attr);
    
    if strcmp(mode, 'vertical') | strcmp(mode, 'v')
        x = width - left - offset;
        y = height / 2;
        
        % fraction of space available
        width  = fshort * left;
        length = flong * height;
        hor = '';
    elseif strcmp(mode, 'horizontal') | strcmp(mode, 'h')
        x = width / 2;
        y = bottom - offset;
        
        % fraction of space available
        length  = flong * width;
        width   = fshort * bottom;
        hor = 'h';
    else
        error('mode should be either: ''vertical'', ''horizontal'', ''v'' or ''h''');
    end
    
    out.x = [num2str(x), 'p'];
    out.y = [num2str(y), 'p'];
    out.length = [num2str(length), 'p'];
    out.width = [num2str(width), 'p', hor];
end

function Gmt = colorbar(Gmt, varargin)

    args = struct('mode', 'v', 'offset', 100, 'flong', 0.8, 'fshort', 0.2, ...
                  'flags', '');
    args = parseArgs(varargin, args);
    
    mode    = args.mode;
    offset  = args.offset;
    flong   = args.flong;
    fshort  = args.fshort;
    
    klass = 'numeric';
    attr = {'scalar', 'positive', 'nonnan', 'real', 'finite'};
    
    validateattributes(mode, {'char'}, {'nonempty'});
    validateattributes(flags, {'char'}, {});
    validateattributes(offset, klass, attr);
    validateattributes(flong, klass, attr);
    validateattributes(fshort, klass, attr);
    
    out = scale_pos(mode, 'offset', offset, 'flong', flong, 'fshort', fshort);
    
    Cmd = sprintf('psscale -D0.0/0.0/%g/%g Xf%g Yf%g %s', out.length, ...
                  out.width, out.x, out.y, flags);
    Gmt = gmt_cmd(Cmd, Gmt);
end

function info(data, **flags):
    gmt_flags = '';
    
    if ischar(data) & isfile(data)
        gmt_flags = data;
    else:
        error('data is not a path to an existing file!');
    end

    # if we have flags parse them
    if len(flags) > 0:
        gmt_flags += " ".join(["-{}{}".format(key, proc_flag(flag))
                               for key, flag in flags.items()])
    
    if get_version() > _gmt_five:
        Cmd = "gmt info " + gmt_flags
    else:
        Cmd = "gmtinfo " + gmt_flags
    
    return cmd(Cmd, ret_out=True).decode()


function out = is_plotter(command)
    plotters = {'grdcontour', 'grdimage', 'grdvector', 'grdview', 'psbasemap', ...
                'psclip', 'pscoast', 'pscontour', 'pshistogram', 'psimage', ...
                'pslegend', 'psmask', 'psrose', 'psscale', 'pstext', 'pswiggle', ...
                'psxy', 'psxyz', 'gmtlogo'};
    
    split = strsplit(command, ' ');
    
    if ismember(split{1}, plotters)
        out = true;
    else
        out = false;
    end
end

function out = cmd(command)
    
    [status, out] = system(command);
    
    if status ~= 0
        str = sprintf('Command (%s) nonzero return status: %d\nOutput: %s', ...
                       command, status, out);
        error(str);
    end
end

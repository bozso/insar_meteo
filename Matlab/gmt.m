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
            otherwise
                error(['Unknown function ', fun]);
        end
    end
end

function out = init(outfile)
    
    validateattributes(outfile, {'char'}, {'nonempty'});
    
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
    out.outfile = outfile;
    % sentinel
    out.commands = 'init';
end

function Gmt = gmt_cmd(Cmd, Gmt)
    Gmt.commands = strjoin({Gmt.commands, Cmd}, '\n');
end

function finalize(Gmt)
    commands = strsplit(Gmt.commands, '\n');
    outfile = Gmt.outfile;
    
    if ~strcmp(commands{1}, 'init')
        error('Gmt struct was not initialized correctly!');
    end
    
    ncom = length(commands);
    
    idx = [];
    
    % get the indices of plotter functions
    for ii = 2:length(commands)
        if is_plotter(commands{ii})
            idx(end + 1) = ii;
        end
    end
    
    % handle -O and -K options
    if numel(idx) > 1
        
        for ii = idx(1:end - 1)
            commands{ii} = [commands{ii}, ' -O'];
        end

        first = idx(1);
        commands{first} = [commands{first}, ' > ', outfile];
        
        for ii = idx(2:end)
            commands{ii} = [commands{ii}, ' -K >> ', outfile];
        end
    end

    if Gmt.is_five
        for ii = 2:ncom
            commands{ii} = ['gmt ', commands{ii}];
        end
    end
    
    commands
    
    %for ii = 2:ncom
    %    cmd(commands{ii});
    %end
end

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

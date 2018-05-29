classdef Gmt < handle
    properties
        version
        is_five
        psfile
        common
        debug
        
        left
        right
        top
        bottom
        
        commands
        outfiles
    end
    
    methods
            %gmt_cmds = {'grdcontour', 'grdimage', 'grdvector', 'grdview', 'psbasemap', ...
            %'psclip', 'pscoast', 'pscontour', 'pshistogram', 'psimage', 'pslegend', ...
            %'psmask', 'psrose', 'psscale', 'pstext', 'pswiggle', 'psxy', 'psxyz', ...
            %'gmtlogo', 'blockmean', 'blockmedian', 'blockmode', 'filter1d', ...
            %'fitcircle', 'gmt2kml', 'gmt5syntax', 'gmtconnect', 'gmtconvert', ...
            %'gmtdefaults', 'gmtget', 'gmtinfo', 'gmtlogo', 'gmtmath', 'gmtregress', ...
            %'gmtselect', 'gmtset', 'gmtsimplify', 'gmtspatial', 'gmtvector', ...
            %'gmtwhich', 'grd2cpt', 'grd2rgb', 'grd2xyz', 'grdblend', 'grdclip', ...
            %'grdcontour', 'grdconvert', 'grdcut', 'grdedit', 'grdfft', 'grdfilter', ...
            %'grdgradient', 'grdhisteq', 'grdimage', 'grdinfo', 'grdlandmask', ...
            %'grdmask', 'grdmath', 'grdpaste', 'grdproject', 'grdraster', 'grdsample', ...
            %'grdtrack', 'grdtrend', 'grdvector', 'grdview', 'grdvolume', 'greenspline', ...
            %'kml2gmt', 'mapproject', 'nearneighbor', 'project', 'sample1d', ...
            %'spectrum1d', 'sph2grd', 'sphdistance', 'sphinterpolate', 'sphtriangulate', ...
            %'splitxyz', 'surface', 'trend1d', 'trend2d', 'triangulate', 'xyz2grd'};
            
        function obj = Gmt(psfile, varargin)
            validateattributes(psfile, {'char'}, {'nonempty'});
            
            args = struct('common', '', 'left', 50, 'right', 50, 'top', 25, 'bottom', 50, ...
                          'debug', false);
            args = Staux.parse_args(varargin, args, {'debug'});
            
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
            
            version = Gmt.cmd('gmt --version');
            
            if strcmp(version(1), '5')
                is_five = true;
            elseif strcmp(version(1), '6')
                error('Cannot handle GMT 6.x!');
            elseif strcmp(version(1), '4')
                is_five = false;
            else
                error('GMT version is not 4,5 or 6!');
            end
            
            obj.version = version;
            obj.is_five = is_five;
            obj.psfile = psfile;
            obj.common = common;
            obj.debug = args.debug;
            
            obj.left = left;
            obj.right = right;
            obj.top = top;
            obj.bottom = bottom;
            
            % sentinels
            obj.commands = 'init';
            obj.outfiles = 'init';
        end % init
        
        function [] = finalize(obj)
            commands = strsplit(obj.commands, '\n');
            outfiles = strsplit(obj.outfiles, '\n');
            common = obj.common;
            
            if ~strcmp(commands{1}, 'init') | ~strcmp(outfiles{1}, 'init')
                error('Gmt struct was not initialized correctly!');
            end
            
            ncom = length(commands);
            
            idx = [];
            
            % get the indices of plotter functions
            for ii = 2:ncom
                if Gmt.is_plotter(commands{ii})
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
                    commands{ii} = [commands{ii}, ' -K'];
                end
                
                first = idx(1);
                commands{first} = [commands{first}, ' > ', outfiles{first}];
                
                for ii = idx(2:end)
                    commands{ii} = [commands{ii}, ' -O >> ', outfiles{ii}];
                end
            else
                first = idx(1);
                commands{first} = [commands{first}, ' > ', outfiles{first}];
            end
            
            % append 'gmt ' to the command strings
            if obj.is_five
                for ii = 2:ncom
                    commands{ii} = ['gmt ', commands{ii}];
                end
            end
            
            if obj.debug
                fprintf('DEBUG: GMT commands\n%s\n', strjoin(commands, '\n'));
            end
            
            for ii = 2:ncom
                fid = Staux.sfopen(outfiles{ii}, 'w');
                fclose(fid);
            end
            
            for ii = 2:ncom
                Gmt.cmd(commands{ii});
            end
        end % finalize
        
        function [x, y] = multiplot(obj, nplots, proj, varargin)
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
            args = Staux.parse_args(varargin, args);
            
            xpad = args.xpad;
            ypad = args.ypad;
            
            klass = 'numeric';
            attr = {'scalar', 'positive', 'nonnan', 'real', 'finite'};
            validateattributes(xpad, klass, attr);
            validateattributes(ypad, klass, attr);
            
            nrows = ceil(sqrt(nplots) - 1);
            nrows = max(1, nrows);
            ncols = ceil(nplots / nrows),
            
            width = get_width(gmt);
        
            % width available for plotting
            awidth = width - (left + right)
                
            % width of a single plot
            pwidth  = (awidth - (ncols - 1) * xpad) / ncols;
                
            obj.common = sprintf('%s -J%g%gp', Gmt.common, proj, pwidth);
                
            % height of a single plot
            pheight = get_height(gmt);
            
            x = left:(pwidth + xpad):(left + ncols * (pwidth + x_pad));
            y = (height - top - pheight - y_pad):-(pheight + ypad):...
                (height - top - nrows * (pheight + ypad));
                
            %# calculate psbasemap shifts in x and y directions
            %x = (left + ii * (pwidth + x_pad) for jj in range(nrows)
                                              %for ii in range(ncols))
            
            %y = (height - top - ii * (pheight + y_pad)
                 %for ii in range(1, nrows + 1)
                 %for jj in range(ncols))
            
            % residual margin left at the bottom
            obj.bottom = height - top - nrows * (pheight + ypad);
        end % multiplot
        
        function [] = call(obj, varargin)
            if nargin == 2
                Cmd = varargin{1};
                outfile = obj.psfile;
            elseif nargin == 3
                Cmd = varargin{1};
                outfile = varargin{2};
            else
                error('1 or 2 arguments required!');
            end
            
            obj.commands = sprintf('%s\n%s', obj.commands, Cmd);
            obj.outfiles = sprintf('%s\n%s', obj.outfiles, outfile);
        end
        
        function width = get_width(obj)
            
            if ismac
                null = '/dev/null';
            elseif isunix
                null = '/dev/null';
            elseif ispc
                null = 'NUL';
            else
                error('Could not identify platform!');
            end
            
            version = Gmt.cmd('gmt --version');
            
            if obj.is_five
                Cmd = sprintf('gmt mapproject %s -Dp', obj.common);
            else
                Cmd = sprintf('mapproject %s -Dp', obj.common);
            end
            
            % before version 5.2
            if version(3) <= '1'
                Cmd = sprintf('%s %s -V', Cmd, null);
                out = strsplit(Gmt.cmd(Cmd), '\n');
                
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
                width = str2num(outsplit{5});
            else
                Cmd = sprintf('%s -Ww', Cmd);
                width = str2num(Gmt.cmd(Cmd));
            end
        end % get_width
        
        function height = get_height(obj)
            
            if ismac
                null = '/dev/null';
            elseif isunix
                null = '/dev/null';
            elseif ispc
                null = 'NUL';
            else
                error('Could not identify platform!');
            end
            
            version = Gmt.cmd('gmt --version');
            
            if obj.is_five
                Cmd = sprintf('gmt mapproject %s -Dp', obj.common);
            else
                Cmd = sprintf('mapproject %s -Dp', obj.common);
            end
            
            % before version 5.2
            if version(3) <= '1'
                Cmd = sprintf('%s %s -V', Cmd, null);
                out = strsplit(Gmt.cmd(Cmd), '\n');
                
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
                outsplit = strsplit(outsplit{7});
                height = str2num(outsplit{1});
            else
                Cmd = sprintf('%s -Wh', Cmd);
                height = str2num(Gmt.cmd(Cmd));
            end
        end % get_height
        
        function [x, y, width, length] = scale_pos(obj, mode, varargin)
            
            args = struct('offset', 100, 'flong', 0.8, 'fshort', 0.2);
            args = Staux.parse_args(varargin, args);
            
            left    = obj.left;
            right   = obj.right;
            top     = obj.top;
            bottom  = obj.bottom;
        
            width   = get_width(obj);
            height  = get_height(obj);
            
            offset  = args.offset;
            flong   = args.flong;
            fshort  = args.fshort;
        
            klass = {'numeric'};
            attr = {'scalar', 'positive', 'nonnan', 'real', 'finite'};
            
            validateattributes(offset, klass, attr);
            validateattributes(flong, klass, attr);
            validateattributes(fshort, klass, attr);
            
            if strcmp(mode, 'vertical') | strcmp(mode, 'v')
                x = width - left + offset;
                y = height / 2;
                
                % fraction of space available
                width  = fshort * left;
                length = flong * height;
                hor = '';
            elseif strcmp(mode, 'horizontal') | strcmp(mode, 'h')
                x = width / 2;
                y = bottom + offset;
                
                % fraction of space available
                length  = flong * width;
                width   = fshort * bottom;
                hor = 'h';
            else
                error('mode should be either: ''vertical'', ''horizontal'', ''v'' or ''h''');
            end
            
            x       = sprintf('%gp', x);
            y       = sprintf('%gp', y);
            length  = sprintf('%gp', length);
            width   = sprintf('%gp%s', width, hor);
        end % scale_pos
        
        function gmt = colorbar(obj, varargin)
        
            args = struct('mode', 'v', 'offset', 100, 'flong', 0.8, 'fshort', 0.2, ...
                          'flags', '');
            args = Staux.parse_args(varargin, args);
            
            mode    = args.mode;
            offset  = args.offset;
            flong   = args.flong;
            fshort  = args.fshort;
            flags   = args.flags;
            
            klass = {'numeric'};
            attr = {'scalar', 'positive', 'nonnan', 'real', 'finite'};
            
            validateattributes(mode, {'char'}, {'nonempty'});
            validateattributes(flags, {'char'}, {});
            validateattributes(offset, klass, attr);
            validateattributes(flong, klass, attr);
            validateattributes(fshort, klass, attr);
            
            [x, y, width, length] = scale_pos(obj, mode, 'offset', offset, ...
                                              'flong', flong, 'fshort', fshort);
            
            Cmd = sprintf('psscale -D0.0/0.0/%s/%s -Xf%s -Yf%s %s', length, ...
                          width, x, y, flags);
            call(obj, Cmd, gmt);
        end % colorbar
    end % methods
    
    methods(Static)
        function [] = makecpt(flags, outfile)
            if obj.is_five
                outs = Gmt.cmd(sprintf('gmt makecpt %s > %s', flags, outfile));
            else
                outs = Gmt.cmd(sprintf('makecpt %s > %s', flags, outfile));
            end
        end

        function out = get_version()
            out = Gmt.cmd('gmt --version');
        end
        
        function out = info(data, flags)
            version = Gmt.cmd('gmt --version');
            
            if ischar(data) & exist(data, 'file') == 2
                gmt_flags = data;
            else
                error('data is not a path to an existing file!');
            end
            
            % if we have flags parse them
            if nargin == 2
                gmt_flags = sprintf('%s %s', gmt_flags, flags);
            end
            
            if version(1) == '5'
                Cmd = ['gmt info ', gmt_flags];
            else
                Cmd = ['gmtinfo ', gmt_flags];
            end
            
            out = Gmt.cmd(Cmd);
        end % info
        
        function [xy_range, z_range] = get_ranges(data, varargin)
            
            args = struct('binary', '', 'xy_add', nan, 'z_add', nan, ...
                          'flags', '');
            args = Staux.parse_args(varargin, args);
            
            binary = args.binary;
            xy_add = args.xy_add;
            z_add  = args.z_add;
            flags  = args.flags;
            
            validateattributes(binary, {'char'}, {});
            validateattributes(flags,  {'char'}, {});
        
            klass = {'numeric'};
            attr = {'scalar', 'positive', 'real'};
            validateattributes(xy_add, klass, attr);
            validateattributes(z_add, klass, attr);
            
            if isempty(binary)
                ranges = str2num(info(data, sprintf('-C %s', flags)));
            else
                ranges = str2num(info(data, sprintf('-bi%s -C %s', binary, flags)));
            end
            
            if ~isnan(xy_add)
                X = (ranges(2) - ranges(1)) * xy_add;
                Y = (ranges(4) - ranges(3)) * xy_add;
                xy_range = [ranges(1) - xy_add, ranges(2) + xy_add,
                            ranges(3) - xy_add, ranges(4) + xy_add];
            else
                xy_range = ranges(1:4);
            end
            
            non_xy = ranges(5:end);
            
            if ~isnan(z_add)
                min_z = min(non_xy); max_z = max(non_xy);
                Z = (max_z - min_z) * z_add;
                z_range = [min_z - z_add, max_z + z_add];
            else
                z_range = [min(non_xy), max(non_xy)];
            end
        end % get_ranges
        
        function [] = plot_scatter(scatter_file, ncols, ps_file, varargin)
            
            args = struct('proj', 'M', 'idx', [], 'config', {}, 'offset', 25, ...
                          'mode', 'v', 'label', '', 'z_range', [], 'xy_range', [], ...
                          'x_axis', 'a0.5g0.25f0.25', 'y_axis', 'a0.25g0.25f0.25', ...
                          'xy_add', 0.05, 'z_add', 0.1, 'colorscale', 'drywet', ...
                          'tryaxis', 0, 'left', 50, 'right', 100, 'top', 0, ...
                          'titles', [], 'point_style', 'c0.25c');
            args = Staux.parse_args(varargin, args, {'tryaxis'});
            
            proj        = args.proj;
            idx         = args.idx;
            config      = args.config;
            mode        = args.mode;
            label       = args.label;
            z_range     = args.z_range;
            xy_range    = args.xy_range;
            xy_add      = args.xy_add;
            z_add       = args.z_add;
            x_axis      = args.x_axis;
            y_axis      = args.y_axis;
            colorscale  = args.colorscale;
            offset      = args.offset;
            point_style = args.point_style;
            
            left    = args.left;
            right   = args.right;
            top     = args.top;
            
            titles = args.titles;
            
            validateattributes(proj, {'char'}, {'nonempty'});
            validateattributes(x_axis, {'char'}, {'nonempty'});
            validateattributes(y_axis, {'char'}, {'nonempty'});
            validateattributes(colorscale, {'char'}, {'nonempty'});
            validateattributes(point_style, {'char'}, {'nonempty'});
        
            klass = 'numeric';
            attr = {'scalar', 'positive', 'real', 'finite', 'nonnan'};
        
            validateattributes(xy_add, klass, attr);
            validateattributes(z_add, klass, attr);
            validateattributes(offset, klass, attr);
            
            validateattributes(left, klass, attr);
            validateattributes(rigth, klass, attr);
            validateattributes(top, klass, attr);
            
            % 2 additional coloumns for coordinates
            bindef = sprintf('%dd', ncols + 2);
            
            if isempty(xy_range) | isempty(z_range)
                out = get_ranges(scatter_file, 'binary', bindef, 'xy_add', xy_add, ...
                                 'z_add', z_add);
            end
            
            if isempty(xy_range)
                xy_range = out.xy_range;
            end
            
            if isempty(z_range)
                z_range = out.z_range;
            end
            
            if isempty(idx)
                idx = 1:ncols;
            end
            
            if isempty(titles)
                titles = idx;
            end
            
            gmt = Gmt(ps_file, 'common', sprintf('-R%s', Gmt.arr2str(xy_range)), ...
                      'config', config);
            
            gmt.multiplot(numel(idx), proj, 'right', right, ...
                          'top', top, 'left', left);
            
            return
            
            Gmt.makecpt(sprintf('-C%s -Z -T%s', colorscale, ...
                        Gmt.arr2str(z_range), 'tmp.cpt'));
            
            for ii = idx
                input_format = sprintf('0,1,%d', ii + 2);
                
                Cmd = sprintf('psbasemap  Xf%gp Yf%gp BWSen+t%s Bx%s By%s', ...
                              x(ii), y(ii), titles(ii), x_axis, y_axis);
                gmt.call(Cmd);
        
                % do not plot the scatter points yet just see the placement of
                % basemaps
                if ~args.tryaxis
                    Cmd = sprintf('psxy %s -i0,1,%d bi%s Sc0.025c Ctmp.cpt', ...
                                  scatter_file, ii + 2);
                    gmt.call(Cmd);
                end
            end
            
            gmt.colorbar('mode', mode, 'offset', offset, 'label', label);
        
            delete('tmp.cpt');
        end % plot_scatter
        
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
                str = sprintf('Command (%s) nonzero return status: %d\n%s', ...
                               command, status, out);
                error(str);
            end
            
            outs = strsplit(out, '\n');
            
            out = {};
            
            % filter out gmt warning/error messages
            for ii = 1:length(outs)
                if isempty(strfind(outs{ii}, 'gmt:'))
                    out{end+1} = outs{ii};
                end
            end
            
            out = strjoin(out, '\n');
        end
        
        function out = arr2str(array)
            tmp = sprintf('%g/', array);
            out = tmp(1:end-1);
        end
    end % methods
end % Gmt

classdef Daisy
    methods(Static)
        function [] = gmtfiles(varargin)
        
            args = struct('scalex', 1.0, 'scaley', 1.0);
            args = parseArgs(varargin, args);
            
            scalex = args.scalex;
            scaley = args.scaley;
            
            attr = {'scalar', 'positive', 'finite', 'nonnan'};
            validateattributes(scalex, {'numeric'}, attr);
            validateattributes(scaley, {'numeric'}, attr);
            
            data = load('integrate.xyi', '-ascii');
            ndata = size(data, 1);
            
            staux('save_ascii', [data(:,1:2), zeros(ndata, 1), data(:,4) .* scalex], ...
                  'integrate_eastwest.xy', '%f');
            staux('save_ascii', [data(:,1:2), repmat(90.0, ndata, 1), ...
                  data(:,5) .* scaley], 'integrate_up.xy', '%f');
        end
        
        function [] = steps(start, stop, varargin)
            
            args = struct('asc_data', 'asc_data.xy', 'dsc_data', 'dsc_data.xy', ...
                          'asc_orbit', 'asc_master.res', 'dsc_orbit', 'dsc_orbit.res', ...
                          'ps_sep', 100.0, 'poly_deg', 3);
            args = parseArgs(varargin, args);
            
            asc_data = args.asc_data;
            dsc_data = args.dsc_data;
            
            asc_orbit = args.asc_orbit;
            dsc_orbit = args.dsc_orbit;
            
            ps_sep = args.ps_sep;
            poly_deg = args.poly_deg;
            
            validateattributes(asc_data, {'char'}, {'nonempty'});
            validateattributes(dsc_data, {'char'}, {'nonempty'});
            
            validateattributes(asc_orbit, {'char'}, {'nonempty'});
            validateattributes(dsc_orbit, {'char'}, {'nonempty'});
            
            validateattributes(ps_sep, {'numeric'}, {'scalar', 'positive', 'real', ...
                               'finite', 'nonnan'});
            validateattributes(poly_deg, {'numeric'}, {'scalar', 'positive', 'real', ...
                               'finite', 'nonnan', 'intiger'});
            
            if start == 1
                data_select(asc_data, dsc_data, ps_sep);
            end
            
            if start >= 2 & stop <= 2
                dominant(ps_sep);
            end
            
            if start >= 3 & stop <= 3
                poly_orbit(asc_orbit, deg);
                poly_orbit(dsc_orbit, deg);
            end
            
            if stop == 4
                integrate();
            end
        end
        
        function [] = data_select(asc_data, dsc_data, ps_sep)
            
            if nargin < 3
                ps_sep = 100.0
            elseif nargin > 3
                error('Too many arguments!');
            end
            
            cmd(sprintf('daisy data_select %s %s %f', asc_data, dsc_data, ps_sep));
        
        function [] = dominant(ps_sep)
        
            if nargin < 1
                ps_sep = 100.0
            elseif nargin > 1
                error('Too many arguments!');
            end
            
            cmd(sprintf('daisy dominant asc_data.xys dsc_data.xys %f', ps_sep));
            
        function [] = poly_orbit(orbit_file, deg)
        
            if nargin < 2
                deg = 3;
            elseif nargin > 2
                error('Too many arguments!');
            end
            
            cmd(sprintf('daisy poly_orbit %s %d', orbit_file, ps_sep));
        end
        
        function [] = integrate()
            cmd('daisy integrate dominant.xyd asc_master.porb dsc_master.porb');
        end
        
        function out = cmd(command)
            
            [status, out] = system(command);
        
            if status ~= 0
                str = sprintf('Command (%s) nonzero return status: %d\n%s', ...
                               command, status, out);
                error(str);
            end
        end
    end % methods
end % Daisy

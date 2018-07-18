% INMET
% Copyright (C) 2018  MTA CSFK GGI
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

classdef Daisy
    methods(Static)
        function [] = gmtfiles(scale)
        
            if nargin < 1
                scale = 1.0
            elseif nargin > 1
                error('Too many arguments!');
            end
            
            attr = {'scalar', 'positive', 'finite', 'nonnan'};
            validateattributes(scale, {'numeric'}, attr);
            
            data = load('integrate.xyi', '-ascii');
            ndata = size(data, 1);
            
            Staux.save_ascii([data(:,1:2), zeros(ndata, 1), data(:,4) .* scale], ...
                             'integrate_eastwest.xy', '%f');
            Staux.save_ascii([data(:,1:2), data(:,5)], 'integrate_up.xy', '%f');
        end
        
        function [] = steps(start, stop, varargin)
            
            args = struct('asc_data', 'asc_data.xy', 'dsc_data', 'dsc_data.xy', ...
                          'asc_orbit', 'asc_master.res', 'dsc_orbit', 'dsc_orbit.res', ...
                          'ps_sep', 100.0, 'poly_deg', 3);
            args = Staux.parse_args(varargin, args);
            
            asc_data = args.asc_data;
            dsc_data = args.dsc_data;
            
            asc_orbit = args.asc_orbit;
            dsc_orbit = args.dsc_orbit;
            
            ps_sep = args.ps_sep;
            poly_deg = args.poly_deg;
            
            klass = {'char'};
            validateattributes(asc_data, klass, {'nonempty'});
            validateattributes(dsc_data, klass, {'nonempty'});
            
            validateattributes(asc_orbit, klass, {'nonempty'});
            validateattributes(dsc_orbit, klass, {'nonempty'});
            
            klass = {'numeric'};
            validateattributes(ps_sep, klass, {'scalar', 'positive', ...
                               'real', 'finite', 'nonnan'});
            validateattributes(poly_deg, klass, {'scalar', 'positive', ...
                               'real', 'finite', 'nonnan', 'integer'});
            
            if start == 1
                Daisy.data_select(asc_data, dsc_data, ps_sep);
            end
            
            if start >= 2 & stop <= 2
                Daisy.dominant(ps_sep);
            end
            
            if start >= 3 & stop <= 3
                Daisy.poly_orbit(asc_orbit, deg);
                Daisy.poly_orbit(dsc_orbit, deg);
            end
            
            if stop == 4
                Daisy.integrate();
            end
        end
        
        function [] = data_select(asc_data, dsc_data, ps_sep)
            
            if nargin < 3
                ps_sep = 100.0
            elseif nargin > 3
                error('Too many arguments!');
            end
            
            Daisy.cmd(sprintf('daisy data_select %s %s %f', asc_data, dsc_data, ...
                        ps_sep))
        end
        
        function [] = dominant(ps_sep)
        
            if nargin < 1
                ps_sep = 100.0
            elseif nargin > 1
                error('Too many arguments!');
            end
            
            Daisy.cmd(sprintf('daisy dominant asc_data.xys dsc_data.xys %f', ...
                      ps_sep))
        end
            
        function [] = poly_orbit(orbit_file, deg)
        
            if nargin < 2
                deg = 3;
            elseif nargin > 2
                error('Too many arguments!');
            end
            
            Daisy.cmd(sprintf('daisy poly_orbit %s %d', orbit_file, ps_sep))
        end
        
        function [] = integrate()
            Daisy.cmd(['daisy integrate dominant.xyd asc_master.porb ', ...
                       'dsc_master.porb'])
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

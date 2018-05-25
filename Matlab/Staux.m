classdef Staux
    methods(Static)
        function ArgStruct = parse_args(args, ArgStruct, varargin)
        % Helper function for parsing varargin. 
        %
        %
        % ArgStruct = parse_args(varargin, ArgStruct [, FlagtypeParams [, Aliases]])
        %
        % * ArgStruct is the structure full of named arguments with default values.
        % * Flagtype params is params that don't require a value. 
        %   (the value will be set to 1 if it is present)
        % * Aliases can be used to map one argument-name to several argstruct fields
        %
        %
        % example usage: 
        % --------------
        % function parseargtest(varargin)
        %
        % %define the acceptable named arguments and assign default values
        % Args = struct('Holdaxis', 0, ...
        %        'SpacingVertical', 0.05, 'SpacingHorizontal', 0.05, ...
        %        'PaddingLeft', 0, 'PaddingRight', 0, 'PaddingTop', 0, 'PaddingBottom', 0, ...
        %        'MarginLeft', 0.1, 'MarginRight', 0.1, 'MarginTop', 0.1, 'MarginBottom', 0.1, ...
        %        'rows', [], 'cols', []); 
        %
        % %The capital letters define abrreviations.  
        % %  Eg. parseargtest('spacingvertical', 0) is equivalent to parseargtest('sv',0) 
        %
        % Args=parseArgs(varargin,Args, ... % fill the arg-struct with values entered by the user
        %           {'Holdaxis'}, ... %this argument has no value (flag-type)
        %           {'Spacing' {'sh','sv'}; 'Padding' {'pl','pr','pt','pb'}; ...
        %            'Margin' {'ml','mr','mt','mb'}});
        %
        % disp(Args)
        %
        %
        %
        %
        % Aslak Grinsted 2004
        
        % -------------------------------------------------------------------------
        %   Copyright (C) 2002-2004, Aslak Grinsted
        %   This software may be used, copied, or redistributed as long as it is not
        %   sold and this copyright notice is reproduced on each copy made.  This
        %   routine is provided as is without any express or implied warranties
        %   whatsoever.
        
            % if we do not have arguments return - added by István Bozsó 2018.05.20.
            if length(varargin) == 0
                return
            end
            
            Aliases = {};
            FlagTypeParams = '';
            
            if (length(varargin) > 0) 
                FlagTypeParams = lower(strvcat(varargin{1}));
                if length(varargin) > 1
                    Aliases = varargin{2};
                end
            end
             
            %---------------Get "numeric" arguments
            NumArgCount = 1;
        
            while (NumArgCount <= size(args, 2)) & (~ischar(args{NumArgCount}))
                NumArgCount = NumArgCount + 1;
            end
            
            NumArgCount = NumArgCount - 1;
        
            if (NumArgCount > 0)
                ArgStruct.NumericArguments = {args{1:NumArgCount}};
            else
                ArgStruct.NumericArguments = {};
            end 
            
            %--------------Make an accepted fieldname matrix (case insensitive)
            Fnames = fieldnames(ArgStruct);
        
            for ii = 1:length(Fnames)
                name = lower(Fnames{ii,1});
                
                % col2 = lower
                Fnames{ii,2} = name; 
                AbbrevIdx = find(Fnames{ii,1} ~= name);
                
                %col3=abreviation letters (those that are uppercase in the ArgStruct) 
                % e.g. SpacingHoriz->sh; the space prevents strvcat from removing 
                % empty lines
                Fnames{ii,3} = [name(AbbrevIdx) ' ']; 
                
                % Does this parameter have a value?
                Fnames{ii,4} = isempty(strmatch(Fnames{ii,2}, FlagTypeParams)); 
            end
        
            FnamesFull = strvcat(Fnames{:,2});
            FnamesAbbr = strvcat(Fnames{:,3});
            
            if length(Aliases) > 0  
                for ii = 1:length(Aliases)
                    name = lower(Aliases{ii,1});
                    
                    % try abbreviations (must be exact)
                    FieldIdx = strmatch(name, FnamesAbbr, 'exact'); 
                    if isempty(FieldIdx) 
                        % &??????? exact or not? 
                        FieldIdx = strmatch(name, FnamesFull); 
                    end
                    Aliases{ii,2} = FieldIdx;
                    AbbrevIdx = find(Aliases{ii,1} ~= name);
                    
                    %the space prevents strvcat from removing empty lines
                    Aliases{ii,3} = [name(AbbrevIdx) ' ']; 
                    % dont need the name in uppercase anymore for aliases
                    Aliases{ii,1} = name; 
                end
                % Append aliases to the end of FnamesFull and FnamesAbbr
                FnamesFull = strvcat(FnamesFull, strvcat(Aliases{:,1})); 
                FnamesAbbr = strvcat(FnamesAbbr, strvcat(Aliases{:,3}));
            end
            
            %--------------get parameters--------------------
            l = NumArgCount + 1; 
            while (l <= length(args))
                a = args{l};
                if ischar(a)
                    % assume that the parameter has is of type 'param',value
                    paramHasValue = 1; 
                    a = lower(a);
                    % try abbreviations (must be exact)
                    FieldIdx = strmatch(a, FnamesAbbr, 'exact'); 
                    
                    if isempty(FieldIdx) 
                        FieldIdx = strmatch(a, FnamesFull); 
                    end
                    
                    % shortest fieldname should win 
                    if (length(FieldIdx) > 1) 
                        [mx, mxi] = max(sum(FnamesFull(FieldIdx,:) == ' ', 2));
                        FieldIdx = FieldIdx(mxi);
                    end
                    
                    % then it's an alias type.
                    if FieldIdx > length(Fnames) 
                        FieldIdx = Aliases{FieldIdx - length(Fnames), 2}; 
                    end
                    
                    if isempty(FieldIdx) 
                        error(['Unknown named parameter: ' a]);
                    end
                    
                    % if it is an alias it could be more than one.
                    for curField=FieldIdx' 
                        if (Fnames{curField, 4})
                            if (l + 1 > length(args))
                                error(['Expected a value for parameter: ' ...
                                       Fnames{curField, 1}]);
                            end
                            val=args{l+1};
                        % FLAG PARAMETER
                        else 
                            % there might be a explicitly specified value for the flag
                            if (l < length(args)) 
                                val = args{l + 1};
                                if isnumeric(val)
                                    if (numel(val) == 1)
                                        val = logical(val);
                                    else
                                        error(['Invalid value for flag-parameter: ' ...
                                               Fnames{curField, 1}])
                                    end
                                else
                                    val = true;
                                    paramHasValue = 0; 
                                end
                            else
                                val = true;
                                paramHasValue = 0; 
                            end
                        end
                        ArgStruct.(Fnames{curField,1}) = val;
                    end
                    % if a wildcard matches more than one
                    l = l + 1 + paramHasValue; 
                else
                    error(['Expected a named parameter: ' num2str(a)]);
                end
            end
        end % parseArgs
        
        function h = boxplot_los(plot_flags, varargin)
        % function h = boxplot_los(plot_flags, 'out', '', 'boxplot_opt', nan, ...
        %                          'fun', nan)
        % 
        % Plots the boxplot of LOS velocities defined by plot_flags.
        % accepted plot flags are the same flags accepted by the ps_plot function,
        % with some extra rules.
        %    1) Multiple plot flags must be defined in a cell array, e.g.
        %    boxplot_los({'v-do', 'v-da'});
        %
        %    2) If we have the atmospheric correction option (''v-da''), the
        %    cooresponding atmospheric correction flag must be defined like this:
        %    'v-da/a_e'. This denotes the DEM error and ERA-I corrected velocity
        %    values. Atmospheric coretcions can be calculated with TRAIN.
        %
        % The plot will be saved to an image file defined by the 'out' argument.
        % No figure will pop up, if the 'out' parameter is defined and is not 
        % an empty string.
        %  
        % Additional options to the boxplot function can be passed using varargin:
        %
        % 'fun' : function to be applied to the velocity values; default value:
        %         nan (no function is applied); function should return a vector
        %         (in the case of a single plot flag) or a matrix
        %         (in the case of multiple plot flags).
        %
        % 'boxplot_opt': varargin arguments for boxplot, given in a cell array;
        %                 e.g.: 'boxplot_opt', {'widths', 0.5, 'whisker', 2.0}
        %                 See the help of the boxplot function for additinal 
        %                 information. Default value: nan (no options)
        % 
        % The function returns the function handle `h` to the boxplot.
            
            args = struct('out', '', 'boxplot_opt', nan, 'fun', nan);
            args = Staux.parse_args(varargin, args);
            
            % loading ps velocities
            vv = Staux.load_ps_vel(plot_flags);
            
            out          = args.out;
            boxplot_opt  = args.boxplot_opt;
            fun          = args.fun;
        
            if isa(fun, 'function_handle')
                vv = fun(vv); % apply the function
            end
        
            % set up labels
            if iscell(boxplot_opt)
                n_var = length(boxplot_opt);
                
                % labels are the velocity flags
                boxplot_opt{n_var + 1} = 'labels';
                boxplot_opt{n_var + 2} = plot_flags;
                
                boxopt = boxplot_opt;
            else
                % labels are the velocity flags
                boxopt{1} = 'labels';
                boxopt{2} = plot_flags;
            end
            
            if isempty(out)
                h = figure;
                boxplot(vv, boxopt{:});
                ylabel('LOS velocity [mm/yr]');
            else
                h = figure('visible', 'off');
                boxplot(vv, boxopt{:});
                ylabel('LOS velocity [mm/yr]');
                saveas(h, out);
            end
        end % boxplot_los
        
        function [] = rel_std_filt(max_rel_std)
        % function rel_std_filt(max_rel_std)
        % 
        % Filters calculated LOS velocities based in their relative standard 
        % deviations.
        % Relative standard deviation = (standard deviation / mean) * 100 
        % (conversion into %).
        % 
        % max_rel_std: maximum allowed realtive standard deviation
        % 
        % Filtered LOS velocities will be saved into "ps_data_filt.xy", in 
        % ascii format.
           
           validateattributes(max_rel_std, {'numeric'}, {'scalar', 'positive', ...
                              'real', 'finite', '<=', 1.0, 'nonnan'});
           
            % create ps_data.xy if it does not exist
            if ~exist('ps_data.xy', 'file')
                ps_output;
            end
            
            if ~exist('ps_mean_v_std.xy', 'file')
                ps_mean_v;
            end
            
            ps_std = load('ps_mean_v_std.xy', '-ascii');
            ps_data = load('ps_data.xy', '-ascii');
            
            rel_std = ps_data(:,3) ./ ps_std(:,3) * 100;
            
            idx = rel_std < max_rel_std;
            
            before = size(ps_data, 1);
            after = sum(idx);
            
            ps_data = ps_data(idx,:);
            
            fprintf(['Number of points before filtering: %d\n', ...
                     'Number of points after filtering: %d\n'], before, after);
            
            save('ps_data_filt.xy', 'ps_data', '-ascii');
        end % rel_std_filt
        
        function [] = iterate_unwrapping(numiter, varargin)
        % function iterate_unwrapping(numiter, 'scla')
        % 
        % Simply iterate the unwrapping process numiter times.
        % At every iteration the spatially-correlated look angle error
        % (StaMPS Step 7) can be calculated.
        % 
        % At the start of the iteration and at every iteration step the phase 
        % residuals will be plotted into a png file, named iteration_(ii).png
        % where ii is the iteration step.
        % 
        % numiter:       (input) number of iteraions
        % 'scla', flag:  (optional) if present SCLA corrections will be 
        %                           calculated and subtracted
            
            validateattributes(numiter, {'numeric'}, {'scalar', 'positive', ...
                               'integer', 'finite', 'nonnan'});
            
            args = struct('scla', 0);
            args = Staux.parse_args(varargin, args, {'scla'});
            
            if ~iscalar(numiter) | numiter < 0.0
                error('numiter should be a positive intiger!');
            end
            
            if args.scla
                end_step = 7;
            else
                end_step = 6;
            end
            
            % remove previous pngs
            delete iteration_*.png;
        
            h = figure('Visible', 'off');
        
            % plot starting residuals
            h = ps_plot('rsb');
            print('-dpng', '-r300', sprintf('iteration_%d.png', 0));
            close(h);
            
            for ii = 1:args.numiter
                fprintf('################\n');
                fprintf('ITERATION #%d\n', ii);
                fprintf('################\n');
                stamps(6, end_step);
                h = figure('Visible', 'off');
                h = ps_plot('rsb');
                print('-dpng', '-r300', sprintf('iteration_%d.png', ii));
                close(h);
            end
        end % iterate_unwrapping
        
        function [] = plot_loop(loop)
        % function plot_loop(loop)
        % 
        % Plots residual phase terms ('rsb') for the selected interferograms.
        % 
        % loop: vector of interferogram indices
        % 
        % E.g.: plot_loop([1 2 3]); will plot 'rsb' values for 
        % IFG 1, 2 and 3.
            validateattributes(loop, {'numeric'}, {'vector', 'intiger', 'finite', ...
                              'nonnan', 'positive'});
        
            ps_plot('rsb', 1, 0, 0, loop);
        end % plot_loop
        
        function out = binned_statistic(x, y, varargin)
        % binned = binned_statistic(x, y, 'bins', 10, 'fun', nan)
        % 
        % Sorts y values into bins defined along x values. By default sums y 
        % values in each of the x bins.
        % 
        % x and y: x and y value pairs, should be a vector with the same 
        % number of elements.
        % 
        % 'bins': optional, number of bins or bin edges defined by a vector.
        % 
        % 'fun': optional, function to apply to y values in each bin.
        % By default this is a summation
        % 
        % E.g. y_binned = binned_statistic(x, y, 'bins', 100, 'fun', @mean)
        % 
        % This will bin y values into x bins and calculate their mean in each 
        % x bins. 100 bins will be placed evenly along the values of x.
            
            klass = {'numeric'};
            attr = {'vector', 'nonempty', 'finite', 'nonnan', 'real'};
            validateattributes(x, klass, attr);
            validateattributes(y, klass, attr);
            
            args = struct('bins', 10, 'fun', nan)
            args = Staux.parse_args(varargin, args);
            
            fun = args.fun;
            bins = args.bins;
            
            if isscalar(bins)
                bins = linspace(min(x), max(x), bins);
            end
            
            % calculate indices that place y values into
            % their respective x bins
            [~, idx] = histc(x, bins);
            
            % do not select values that are out of the range
            % of x bins
            y = y(idx > 0.0);
            idx = idx(idx > 0.0);
            
            if isnan(fun)
                binned = accumarray(idx', y', []);
            else
                binned = accumarray(idx', y', [], fun);
            end
            out.binned = binned;
            out.bins = bins;
        end % binned_statistic
        
        function out = binned_statistic_2d(x, y, z, varargin)
        % binned = binned_statistic_2d(x, y, z, 'xbins', 10, 'ybins', 10, ...
        %                              'fun', nan)
        % 
        % Sorts z values into bins defined along (x,y) values. By default sums 
        % z values in each of the (x,y) bins.
        % 
        % x, y and z: x, y and z value triplets, should be vectors with the 
        % same number of elements.
        % 
        % 'xbins': optional, number of bins or bin edges defined by a vector 
        %                    along x.
        % 
        % 'ybins': optional, number of bins or bin edges defined by a vector 
        %                    along y.
        % 
        % 'fun': optional, function to apply to z values in each bin. 
        % By default this is a summation.
        % 
        % E.g. z_binned = binned_statistic(x, y, z, 'xbins', 100, 'fun', @mean)
        % 
        % This will bin z values into (x,y) bins and calculate their
        % mean in each (x,y) bins. 100 bins will be placed evenly along
        % the values of x and 10 bins along the values of y.
            
            klass = {'numeric'};
            attr = {'vector', 'nonempty', 'finite', 'nonnan', 'real'};
            validateattributes(x, klass, attr);
            validateattributes(y, klass, attr);
            validateattributes(z, klass, attr);
         
            args = struct('xbins', 10, 'ybins', 10, 'fun', nan)
            args = Staux.parse_args(varargin, args);
            
            fun = args.fun;
            xbins = args.xbins;
            ybins = args.ybins;
            
            if isscalar(xbins)
                xbins = linspace(min(x), max(x), xbins);
            end
        
            if isscalar(ybins)
                ybins = linspace(min(y), max(y), ybins);
            end
        
            % calculate indices that place (x,y) values into
            % their respective (x,y) bins
            [~, idx_x] = histc(x, xbins);
            [~, idx_y] = histc(y, ybins);
        
            % do not select values that are out of the range
            % of (x,y) bins
            idx = idx_x > 0.0 & idx_y > 0.0;
            
            z = z(idx);
            idx_x = idx_x(idx);
            idx_y = idx_y(idx);
        
            if strcmp(fun, 'sum')
                binned = accumarray([idx_x, idx_y], z, []);
            else
                binned = accumarray([idx_x, idx_y], z, [], fun);
            end
            out.binned = binned;
            out.xbins = xbins;
            out.ybins = ybins;
        end % binned_statistic_2d
        
        function out = clap(varargin)
        % function out = clap('grid_size', 50, 'alpha', 1, 'beta', 0.3, ...
        %                     'low_pass', 800, 'win_size', 32, 'ifg_list', [])
        % 
        % Based on the Combined Low-pass Adaptive Filter of the StaMPS package
        % developed by Andrew Hooper.
        % Modified CLAP filter. I used it to play around with the filter
        % parameters. Feel free to ingore it.
            
            args = struct('grid_size', 50, 'alpha', 1, 'beta', 0.3, ...
                          'low_pass', 800, 'win_size', 32, 'ifg_list', []);
            
            args = Staux.parse_args(varargin, args);
            
            grid_size           = args.grid_size;
            clap_alpha          = args.alpha;
            clap_beta           = args.beta;
            low_pass_wavelength = args.low_pass;
            n_win               = args.win_size;
            ifg_idx             = args.ifg_list;
            
            scal_pos = {'scalar', 'finite', 'positive', 'nonnan'};
            scal_pos_int = {'scalar', 'finite', 'positive', 'integer', 'nonnan'};
            
            klass = {'numeric'};
            validateattributes(grid_size,           klass, scal_pos);
            validateattributes(clap_alpha,          klass, scal_pos);
            validateattributes(clap_beta,           klass, scal_pos);
            validateattributes(low_pass_wavelength, klass, scal_pos);
            validateattributes(nwin,                klass, scal_pos_int);
            validateattributes(ifg_idx,             klass, scal_pos_int);
        
            freq0 = 1 / low_pass_wavelength;
            freq_i= -n_win / grid_size / n_win / 2:1 / grid_size / ...
                     n_win:(n_win-2) / grid_size / n_win / 2;
            butter_i = 1 ./ (1 + (freq_i / freq0).^(2*5));
            low_pass = butter_i' * butter_i;
            low_pass = fftshift(low_pass);
        
            ps=load('ps1.mat');
            bp=load('bp1.mat');
        
            phin=load('ph1.mat');
            ph=phin.ph;
            clear phin
        
            if isempty(ifg_idx)
                n_ifg = ps.n_ifg;
                ifg_idx = 1:n_ifg;
            elseif isvector(ifg_idx)
                n_ifg = length(ifg_idx);
            else
                n_ifg = 1;
                ifg_idx = [ifg_idx];
            end
                    
            bperp = ps.bperp(ifg_idx);
            n_image = ps.n_image;
            n_ps = ps.n_ps;
            ifgday_ix = ps.ifgday_ix(ifg_idx,:);
            xy = ps.xy;
        
            K_ps = zeros(n_ps,1);
            
            clear ps
        
            xbins = min(xy(:,2)):grid_size:max(xy(:,2));
            ybins = min(xy(:,3)):grid_size:max(xy(:,3));
        
            n_i = length(xbins);
            n_j = length(ybins);
            ph_grid = zeros(n_i, n_j, 'single');
            ph_filt = zeros(n_j, n_i, 'single');
        
            da = load('da1.mat');
            D_A = da.D_A;
            clear da
        
            weighting = 1 ./ D_A;
        
            ph_weight = ph(:,ifg_idx).*exp(-j * bp.bperp_mat(:,ifg_idx).* ...
                        repmat(K_ps, 1, n_ifg)) .* repmat(weighting, 1, n_ifg);
            
            if n_ifg == 1
                ph_grid = binned_statistic_2d(xy(:,2), xy(:,3), ph_weight, ...
                                  'xbins', xbins, 'ybins', ybins);
                ph_filt = clap_filt(transpose(ph_grid), clap_alpha, clap_beta, ...
                                           n_win * 0.75, n_win * 0.25, low_pass);
            else            
                for ii = ifg_idx
                    ph_grid = binned_statistic_2d(xy(:,2), xy(:,3), ph_weight(:,ii), ...
                                      'xbins', xbins, 'ybins', ybins);
                    ph_filt = clap_filt(transpose(ph_grid), clap_alpha, clap_beta, ...
                                               n_win * 0.75, n_win * 0.25, low_pass);
                end
            end
            out.ph_filt = ph_filt;
            out.ph_grid = ph_grid;
        end % clap
        
        
        function h = plot_ph_grid(ph)
        % function h = plot_ph_grid(ph)
        %
        % Auxilliary function for plotting the output of the modified CLAP filter.
        
            attr = {'2d', 'nonempty', 'finite', 'nonnan'};
            validateattributes(ph, {'numeric'}, attr);
            
            h = figure();
            colormap('jet');
            imagesc(angle(ph));
            colorbar();
        end % plot_ph_grid
        
        function vv = load_ps_vel(plot_flags)
        % function vv = load_ps_vel(plot_flags)
        %
        % Helper function that loads LOS velocities defined by the plot_flags cell array.
            
            validateattributes(plot_flags, {'char', 'cell'}, {});
            
            % if we have multiple plot_flags
            if iscell(plot_flags)
                n_flags = length(plot_flags);
        
                ps = load('ps2.mat');
                
                % allocating space for velocity values
                vv = zeros(size(ps.lonlat, 1), n_flags);
                
                clear ps;
                
                for ii = 1:n_flags % going through flags
                    
                    % splitting for atmospheric flags
                    plot_flag = strsplit(plot_flags{ii}, '/');
                    
                    % write velocities into a mat file and load it    
                    
                    % if we have atmospheric flag
                    if length(plot_flag) > 1
                        ps_plot(plot_flag{1}, plot_flag{2}, -1);
                        v = load(sprintf('ps_plot_%s', lower(plot_flag{1})));
                    else
                        ps_plot(plot_flag{1}, -1);
                        v = load(sprintf('ps_plot_%s', lower(plot_flag{1})));
                    end
            
                    % put velocity values into the corresponding column
                    vv(:,ii) = v.ph_disp;
                end % end for
            else
                % splitting for atmospheric flags
                plot_flag = strsplit(plot_flags, '/');
                
                % write velocities into a mat file and load it
                if length(plot_flag) > 1 % if we have atmospheric flag
                    ps_plot(plot_flag{1}, plot_flag{2}, -1);
                    v = load(sprintf('ps_plot_%s', lower(plot_flag{1})));
                else
                    ps_plot(plot_flag{1}, -1);
                    v = load(sprintf('ps_plot_%s', lower(plot_flag{1})));
                end
                vv = v.ph_disp;
            end
        end % load_ps_vel
        
        function h = plot(flag_type, varargin)
        % function h = plot(flag_type, 'out', '', 'background', 1, 'phase_lims', 0, ...
        %                   'ref_ifg', 0, 'n_x', 0, 'cbar_flag', 0, 'textsize', 0, ...
        %                   'textcolor', [], 'lon_rg', [], 'lat_rg', [], 'ts')
        % 
        % Wrapper function for ps_plot with argument handling that is more user 
        % friendly. See tha documentation of ps_plot for arguments.
            
            args = struct('out', '', 'background', 1, 'phase_lims', 0, 'ref_ifg', 0, ...
                          'n_x', 0, 'cbar_flag', 0, 'textsize', 0, 'textcolor', [], ...
                          'lon_rg', [], 'lat_rg', [], 'ts', 0);
            args = Staux.parse_args(varargin, args, {'ts'});
            
            if args.ts
                h = ps_plot(flag_type, 'ts', args.background, args.phase_lims, ...
                            args.ref_ifg, args.ifg_list, args.n_x, args.cbar_flag, ...
                            args.textsize, args.textcolor, args.lon_rg, args.lat_rg);
            else
                h = ps_plot(flag_type, args.background, args.phase_lims, ...
                            args.ref_ifg, args.ifg_list, args.n_x, args.cbar_flag, ...
                            args.textsize, args.textcolor, args.lon_rg, args.lat_rg);
            end
            
            if ~isempty(args.out)
                save(h, args.out);
            end
        end % plot
        
        function [] = report()
        % function [] = report()
        % 
        % Just a bunch of plots, can be safely ignored.
        
            plot('w', 'out', 'wrapped.png');
            plot('u', 'out', 'unwrapped.png');
            plot('u-do', 'out', 'unwrapped_do.png');
            plot('usb', 'out', 'unwrapped_sb.png');
            plot('rsb', 'out', 'rsb.png');
            plot('usb-do', 'out', 'unwrapped_sb_do.png');
            
            plot('V', 'out', 'vel.png');
            plot('Vs', 'out', 'vel_std.png');
            plot('V-do', 'out', 'vel_do.png');
            plot('Vs-do', 'out', 'vel_std_do.png');
        end % report
        
        function [] = output()
        % function [] = output()
        % MODIFIED ps_output from StaMPS. For some reason 
        % save('data.txt', 'data', '-ascii') did not work for me. I made some 
        % simple modifications to make it work with my `save_ascii` 
        % function (see the next function in this library).
        
            %PS_OUTPUT write various output files 
            %
            %   Andy Hooper, June 2006
            %
            %   =======================================================================
            %   09/2009 AH: Correct processing for small baselines output
            %   03/2010 AH: Add velocity standard deviation 
            %   09/2011 AH: Remove code that reduces extreme values
            %   02/2015 AH: Remove code that reduces the extreme values in u-dm
            %   =======================================================================
            
            fprintf('Writing output files...\n')
            
            small_baseline_flag = getparm('small_baseline_flag',1);
            ref_vel = getparm('ref_velocity',1);
            lambda = getparm('lambda',1);
            
            load psver
            psname=['ps', num2str(psver)];
            rcname=['rc', num2str(psver)];
            phuwname=['phuw', num2str(psver)];
            sclaname=['scla', num2str(psver)];
            hgtname=['hgt', num2str(psver)];
            scnname=['scn', num2str(psver)];
            mvname=['mv', num2str(psver)];
            meanvname=['mean_v'];
            
            ps=load(psname);
            phuw=load(phuwname);
            rc=load(rcname);
            
            if strcmpi(small_baseline_flag,'y')
                n_image=ps.n_image;
            else
                n_image=ps.n_ifg;
            end
            
            %ijname=['ps_ij.txt'];
            ij=ps.ij(:,2:3);
            % save(ijname,'ij','-ASCII');
            save_ascii('ps_ij.txt', '%d %d\n', ij)
            
            
            %llname=['ps_ll.txt'];
            lonlat=ps.lonlat;
            % save(llname,'lonlat','-ASCII');
            save_ascii('ps_ll.txt', '%f %f\n', lonlat);
            
            
            %datename=['date.txt'];
            date_out=str2num(datestr(ps.day, 'yyyymmdd'));
            % save(datename,'date_out','-ascii','-double');
            save_ascii('date.txt', '%f\n', date_out);
            
            master_ix = sum(ps.master_day>ps.day) + 1;
            
            ref_ps = ps_setref;
            ph_uw = phuw.ph_uw - repmat(mean(phuw.ph_uw(ref_ps,:)), ps.n_ps,1);
            ph_w = angle(rc.ph_rc.*repmat(conj(sum(rc.ph_rc(ref_ps,:))), ps.n_ps,1));
            ph_w(:,master_ix) = 0;
            
            fid = fopen('ph_w.flt', 'w');
            fwrite(fid, ph_w', 'float');
            fclose(fid);
            
            fid = fopen('ph_uw.flt', 'w');
            fwrite(fid,ph_uw', 'float');
            fclose(fid);
            
            scla = load(sclaname);
        
            if exist([hgtname, '.mat'],'file')
                hgt = load(hgtname);
            else
                hgt.hgt = zeros(ps.n_ps,1);
            end
            
            ph_uw = phuw.ph_uw - scla.ph_scla - repmat(scla.C_ps_uw,1,n_image);
            
            %%% this is only approximate
            K_ps_uw = scla.K_ps_uw-mean(scla.K_ps_uw);
            dem_error = double(K2q(K_ps_uw, ps.ij(:,3)));
            
            hgt_idx = hgt.hgt == 0;
            
            if sum(hgt_idx)
                dem_error = dem_error - mean(dem_error(hgt_idx));
            end
            
            %dem_error=dem_error-mean(dem_error(hgt.hgt==0));
            dem_sort = sort(dem_error);
            min_dem = dem_sort(ceil(length(dem_sort)*0.001));
            max_dem = dem_sort(floor(length(dem_sort)*0.999));
            dem_error_tt = dem_error;
            dem_error_tt(dem_error < min_dem) = min_dem; % for plotting purposes
            dem_error_tt(dem_error>max_dem) = max_dem; % for plotting purposes
            dem_error_tt = [ps.lonlat, dem_error_tt];
            
            % save('dem_error.xy','dem_error_tt','-ascii');
            save_ascii('dem_error.xy', '%f %f %f\n', dem_error_tt);
            
            %%%
            
            clear scla phuw
            ph_uw = ph_uw - repmat(mean(ph_uw(ref_ps,:)), ps.n_ps,1);
            
            meanv = load(meanvname);
            % m(1,:) is master APS + mean deviation from model
            mean_v = - meanv.m(2,:)' * 365.25 / 4 / pi * lambda * 1000 + ref_vel * 1000;
            
            %v_sort=sort(mean_v);
            %min_v=v_sort(ceil(length(v_sort)*0.001));
            %max_v=v_sort(floor(length(v_sort)*0.999));
            %mean_v(mean_v<min_v)=min_v;
            %mean_v(mean_v>max_v)=max_v;
            
            
            %mean_v_name = ['ps_mean_v.xy'];
            mean_v = [ps.lonlat,double(mean_v)];
            %save(mean_v_name,'mean_v','-ascii');
            save_ascii('ps_mean_v.xy', '%f %f %f\n', mean_v);
            
            
            if exist(['./',mvname,'.mat'], 'file');
                mv = load(mvname);
                mean_v_std = mv.mean_v_std;
                v_sort = sort(mean_v_std);
                min_v = v_sort(ceil(length(v_sort)*0.001));
                max_v = v_sort(floor(length(v_sort)*0.999));
                mean_v_std(mean_v_std < min_v) = min_v;
                mean_v_std(mean_v_std > max_v) = max_v;
                mean_v_name = ['ps_mean_v_std.xy'];
                mean_v = [ps.lonlat,double(mean_v_std)];
                %save(mean_v_name,'mean_v','-ascii');
                save_ascii(mean_v_name, '%f %f %f\n', mean_v);
            end
            
            
            %%Note mean_v is relative to a reference point
            %%and dem_error is relative to mean of zero height points (if there are any)
            fid=fopen('ps_data.xy','w');
            fprintf(fid,'%f %f %4.4f %4.4f %4.4f\n',[mean_v,double(hgt.hgt),dem_error]');
            fclose(fid)
            
            for i=1:n_image
                ph=ph_uw(:,i);
        
                ph=-ph*lambda*1000/4/pi;
                ph=[ps.lonlat,double(ph)];
                %save(['ps_u-dm.',num2str(i),'.xy'],'ph','-ascii');
                save_ascii(['ps_u-dm.',num2str(i),'.xy'], '%f %f %f\n', ph);
            end
        
        end % output
        
        function [] = save_ascii(data, path, format)
        % function [] = save_ascii(data, path, format)
        % 
        % Replacement for save(path, 'data', '-ascii')
        %
        % path: filename, where data will be saved.
        % format: second argument of the fprintf function.
        % data: matrix to be saved.
        % 
        % E.g.: save_ascii('data.txt', '%f', ones(5,4));
            
            validateattributes(path, {'char'}, {'nonempty'});
            validateattributes(format, {'char'}, {'nonempty'});
            validateattributes(data, {'numeric'}, {});
            
            ncols = size(data, 2);
            
            format = sprintf('%s\n', repmat(sprintf('%s ', format), 1, ncols));
            
            [FID, msg] = fopen(path, 'w');
            
            if FID == -1
                error(sprintf('Could not open file: %s  Error message: %s', ...
                               path, msg));
            end
        
            fprintf(FID, format, data');
            fclose(FID);
            
        end % save_ascii
        
        function h = plot_scatter(data, varargin)
        % function h = plot_scatter(data, varargin)
        %
        % Function is still under developement final documentation will be created,
        % when the function will be completed.
            
            validateattributes(data, {'numeric'}, {'nonempty', 'finite', ...
                                      'ndims', 2});
        
            args = struct('out', nan, 'psize', 1.0, 'lon_rg', [], 'lat_rg', [], ...
                          'clims', 'auto');
            args = Staux.parse_args(varargin, args)
            
            out    = args.out;
            psize  = args.psize;
            lon_rg = args.lon_rg;
            lat_rg = args.lat_rg;
            clims  = args.clims;
            
            klass = {'numeric'};
            validateattributes(out, {'char'});
            validateattributes(psize, klass, {'scalar', 'positive', 'nonnan', ...
                               'finite'});
            validateattributes(lon_rg, klass, {'vector'});
            validateattributes(lat_rg, klass, {'vector'});
            validateattributes(clims, {'char', 'numeric'});
            
            ncols = size(data, 2);
            
            ps = load('ps2.mat');
            ll = ps.lonlat;
            clear ps;
            
            if ncols == 1
                fcols = 1;
                frows = 1;
            else
                frows = ceil(sqrt(ncols) - 1);
                frows = max(1, frows);
                fcols = ceil(ncols / frows);
            end
            
            if isnan(out)
                h = figure();
            else
                h = figure('visible', 'off');
            end
            
            for ii = 1:ncols
                subplot_tight(frows, fcols, ii);
                scatter(ll(:,1), ll(:,2), psize, data(:,ii));
                caxis(clims);
                colorbar();
            end
            
            if ~isnan(out)
                saveas(h, out);
            end
        end % plot_scatter
        
        function [] = corr_phase(ifg, value)
        % function [] = corr_phase(ifg, value)
        %
        % Add value to the points enclosed by x,y point pair polygons to the 
        % interferogram with index ifg.
            
            klass = {'numeric'};
            validateattributes(ifg, klass, {'scalar', 'intiger', 'finite', ...
                               'nonnan', 'real'});
            validateattributes(value, klass, {'scalar', 'real', 'finite', 'nonnan'});
            
            [x, y] = ginput;
        
            load('phuw_sb2.mat')
            load('ps2.mat')
        
            ph_ifg = ph_uw(:, ifg);
            lon = lonlat(:, 1);
            lat = lonlat(:, 2);
        
            in = inpolygon(lon, lat, x, y);
            
            ph_uw(in,ifg) = ph_uw(in,ifg) + value
        
            save('phuw_sb2.mat', 'ph_uw', 'msd')
        end % corr_phase
        
        function [] = crop(lon_min, lon_max, lat_min, lat_max)
        % function [] = crop(lon_min, lon_max, lat_min, lat_max)
        %
        % Select points in a rectengular area defined by lon_min, lon_max, lat_min, 
        % lat_max. Old *.mat files will be copied (*_old.mat) as a backup.
        % New *.mat files will be created with the selected points.
            
            klass = {'numeric'};
            attr = {'scalar', 'real', 'finite', 'nonnan'};
            
            validateattributes(lon_min, klass, attr);
            validateattributes(lon_max, klass, attr);
            validateattributes(lat_min, klass, attr);
            validateattributes(lat_max, klass, attr);
            
            if ~exist('ps2_old.mat', 'file')
                copyfile ps2.mat ps2_old.mat
            end
        
            if ~exist('pm2_old.mat', 'file')
                copyfile pm2.mat pm2_old.mat
            end
        
            if ~exist('hgt2_old.mat', 'file')
                copyfile hgt2.mat hgt2_old.mat
            end
        
            if ~exist('bp2_old.mat', 'file')
                copyfile bp2.mat bp2_old.mat
            end
        
            if ~exist('rc2_old.mat', 'file')
                copyfile rc2.mat rc2_old.mat
            end
            
            ps = load('ps2_old.mat');
            pm = load('pm2_old.mat');
            bp = load('bp2_old.mat');
            rc = load('rc2_old.mat');
            hgt = load('hgt2_old.mat');
            
            lon = ps.lonlat(:,1);
            lat = ps.lonlat(:,2);
            
            before = size(lon, 1);
            
            idx = ~(lon > lon_min & lon < lon_max & lat > lat_min & lat < lat_max);
            
            after = sum(~idx);
            
            ps.xy(idx,:) = [];
            ps.lonlat(idx,:) = [];
            ps.ij(idx,:) = [];
            
            pm.coh_ps(idx,:) = [];
        
            hgt.hgt(idx,:) = [];
            
            rc.ph_rc(idx,:) = [];
            
            bp.bperp_mat(idx,:) = [];
            
            fprintf('Number of datapoints before cropping: %e and after cropping: %e', ...
                     before, after);
            
            save('hgt2.mat', '-struct', 'hgt');
            save('bp2.mat', '-struct', 'bp');
        
            save('rc2.mat', '-struct', 'rc');
            save('pm2.mat', '-struct', 'pm');
            
            save('ps2.mat', '-struct', 'ps');
        end % crop
        
        function [] = crop_reset()
        % function [] = crop_reset()
        %
        % Reset cropping done with the crop function. See the help of crop.
        
            movefile ps2_old.mat ps2.mat
            movefile pm2_old.mat pm2.mat
            movefile hgt2_old.mat hgt2.mat
            movefile bp2_old.mat bp2.mat
            movefile rc2_old.mat rc2.mat
        end
        
        function [] = save_llh()
            
            ps = load('ps2.mat');
                
            llh = zeros(size(ps.lonlat, 1), 3);
            llh(:,1:2) = ps.lonlat;
            
            clear ps;
            
            hgt = load('hgt2.mat');
            llh(:,3) = hgt.hgt;
            
            clear hgt;
            
            fid = sfopen('llh.dat', 'w');
            fwrite(fid, transpose(llh), 'double');
            fclose(fid);
        end % save_llh
        
        function [] = save_binary(data, path, varargin)
            
            validateattributes(data, {'numeric'}, {'nonempty', 'finite', ...
                                      'ndims', 2, 'nonnan'});
            
            args = struct('dtype', 'double');
            args = Staux.parse_args(varargin, args);
            
            ps = load('ps2.mat');
            
            n_lonlat = size(ps.lonlat, 1);
            
            if size(data, 1) ~= n_lonlat
                error('data should have the same number of rows as lonlat');
            end
            
            ll = zeros(n_lonlat, 2 + size(data, 2));
            ll(:,1:2) = ps.lonlat;
            clear ps;
            
            ll(:,3:end) = data;
            
            fid = sfopen(path, 'w');
            fwrite(fid, transpose(ll), args.dtype);
            fclose(fid);
        end % save_binary
        
        function loaded = load_binary(path, ncols, varargin)
            
            validateattributes(path, {'char'}, {'nonempty'});
            validateattributes(ncols, {'numeric'}, {'scalar', 'intiger', ...
                                       'finite', 'nonnan'});
            
            args = struct('dtype', 'double');
            args = Stuax.parse_args(varargin, args);
            dtype = args.dtype;
            
            validateattributes(dtype, {'char'}, {'nonempty'});
            
            fid = sfopen(path, 'r');
            loaded = transpose(fread(fid, [ncols, Inf], dtype));
            fclose(fid);
        end
        
        function ArgStruct = parseArgs(args, ArgStruct, varargin)
        % Helper function for parsing varargin. 
        %
        %
        % ArgStruct = parseArgs(varargin, ArgStruct [, FlagtypeParams [, Aliases]])
        %
        % * ArgStruct is the structure full of named arguments with default values.
        % * Flagtype params is params that don't require a value. 
        %   (the value will be set to 1 if it is present)
        % * Aliases can be used to map one argument-name to several argstruct fields
        %
        %
        % example usage: 
        % --------------
        % function parseargtest(varargin)
        %
        % %define the acceptable named arguments and assign default values
        % Args = struct('Holdaxis', 0, ...
        %        'SpacingVertical', 0.05, 'SpacingHorizontal', 0.05, ...
        %        'PaddingLeft', 0, 'PaddingRight', 0, 'PaddingTop', 0, 'PaddingBottom', 0, ...
        %        'MarginLeft', 0.1, 'MarginRight', 0.1, 'MarginTop', 0.1, 'MarginBottom', 0.1, ...
        %        'rows', [], 'cols', []); 
        %
        % %The capital letters define abrreviations.  
        % %  Eg. parseargtest('spacingvertical', 0) is equivalent to parseargtest('sv',0) 
        %
        % Args=parseArgs(varargin,Args, ... % fill the arg-struct with values entered by the user
        %           {'Holdaxis'}, ... %this argument has no value (flag-type)
        %           {'Spacing' {'sh','sv'}; 'Padding' {'pl','pr','pt','pb'}; ...
        %            'Margin' {'ml','mr','mt','mb'}});
        %
        % disp(Args)
        %
        %
        %
        %
        % Aslak Grinsted 2004
        
        % -------------------------------------------------------------------------
        %   Copyright (C) 2002-2004, Aslak Grinsted
        %   This software may be used, copied, or redistributed as long as it is not
        %   sold and this copyright notice is reproduced on each copy made.  This
        %   routine is provided as is without any express or implied warranties
        %   whatsoever.
        
            % if we do not have arguments return - added by István Bozsó 2018.05.20.
            if length(varargin) == 0
                return
            end
            
            Aliases = {};
            FlagTypeParams = '';
            
            if (length(varargin) > 0) 
                FlagTypeParams = lower(strvcat(varargin{1}));
                if length(varargin) > 1
                    Aliases = varargin{2};
                end
            end
             
            %---------------Get "numeric" arguments
            NumArgCount = 1;
        
            while (NumArgCount <= size(args, 2)) & (~ischar(args{NumArgCount}))
                NumArgCount = NumArgCount + 1;
            end
            
            NumArgCount = NumArgCount - 1;
        
            if (NumArgCount > 0)
                ArgStruct.NumericArguments = {args{1:NumArgCount}};
            else
                ArgStruct.NumericArguments = {};
            end 
            
            %--------------Make an accepted fieldname matrix (case insensitive)
            Fnames = fieldnames(ArgStruct);
        
            for ii = 1:length(Fnames)
                name = lower(Fnames{ii,1});
                
                % col2 = lower
                Fnames{ii,2} = name; 
                AbbrevIdx = find(Fnames{ii,1} ~= name);
                
                %col3=abreviation letters (those that are uppercase in the ArgStruct) 
                % e.g. SpacingHoriz->sh; the space prevents strvcat from removing 
                % empty lines
                Fnames{ii,3} = [name(AbbrevIdx) ' ']; 
                
                % Does this parameter have a value?
                Fnames{ii,4} = isempty(strmatch(Fnames{ii,2}, FlagTypeParams)); 
            end
        
            FnamesFull = strvcat(Fnames{:,2});
            FnamesAbbr = strvcat(Fnames{:,3});
            
            if length(Aliases) > 0  
                for ii = 1:length(Aliases)
                    name = lower(Aliases{ii,1});
                    
                    % try abbreviations (must be exact)
                    FieldIdx = strmatch(name, FnamesAbbr, 'exact'); 
                    if isempty(FieldIdx) 
                        % &??????? exact or not? 
                        FieldIdx = strmatch(name, FnamesFull); 
                    end
                    Aliases{ii,2} = FieldIdx;
                    AbbrevIdx = find(Aliases{ii,1} ~= name);
                    
                    %the space prevents strvcat from removing empty lines
                    Aliases{ii,3} = [name(AbbrevIdx) ' ']; 
                    % dont need the name in uppercase anymore for aliases
                    Aliases{ii,1} = name; 
                end
                % Append aliases to the end of FnamesFull and FnamesAbbr
                FnamesFull = strvcat(FnamesFull, strvcat(Aliases{:,1})); 
                FnamesAbbr = strvcat(FnamesAbbr, strvcat(Aliases{:,3}));
            end
            
            %--------------get parameters--------------------
            l = NumArgCount + 1; 
            while (l <= length(args))
                a = args{l};
                if ischar(a)
                    % assume that the parameter has is of type 'param',value
                    paramHasValue = 1; 
                    a = lower(a);
                    % try abbreviations (must be exact)
                    FieldIdx = strmatch(a, FnamesAbbr, 'exact'); 
                    
                    if isempty(FieldIdx) 
                        FieldIdx = strmatch(a, FnamesFull); 
                    end
                    
                    % shortest fieldname should win 
                    if (length(FieldIdx) > 1) 
                        [mx, mxi] = max(sum(FnamesFull(FieldIdx,:) == ' ', 2));
                        FieldIdx = FieldIdx(mxi);
                    end
                    
                    % then it's an alias type.
                    if FieldIdx > length(Fnames) 
                        FieldIdx = Aliases{FieldIdx - length(Fnames), 2}; 
                    end
                    
                    if isempty(FieldIdx) 
                        error(['Unknown named parameter: ' a]);
                    end
                    
                    % if it is an alias it could be more than one.
                    for curField=FieldIdx' 
                        if (Fnames{curField, 4})
                            if (l + 1 > length(args))
                                error(['Expected a value for parameter: ' ...
                                       Fnames{curField, 1}]);
                            end
                            val=args{l+1};
                        % FLAG PARAMETER
                        else 
                            % there might be a explicitly specified value for the flag
                            if (l < length(args)) 
                                val = args{l + 1};
                                if isnumeric(val)
                                    if (numel(val) == 1)
                                        val = logical(val);
                                    else
                                        error(['Invalid value for flag-parameter: ' ...
                                               Fnames{curField, 1}])
                                    end
                                else
                                    val = true;
                                    paramHasValue = 0; 
                                end
                            else
                                val = true;
                                paramHasValue = 0; 
                            end
                        end
                        ArgStruct.(Fnames{curField,1}) = val;
                    end
                    % if a wildcard matches more than one
                    l = l + 1 + paramHasValue; 
                else
                    error(['Expected a named parameter: ' num2str(a)]);
                end
            end
        end % parseArgs
        
        function fid = sfopen(path, mode, machine)
        % fid = sfopen(path, mode, machine)
        %
        % A slightly modified fopen function that exits with error if the file defined
        % by path cannot be opened.
        % Accepts the same arguments as fopen.
            
            % filepath to be opened
            if nargin < 1 || isempty(path)
               error('Required argument path is not specified');
            end
            
            % read, write, append
            if nargin < 2 || isempty(mode)
                mode = 'r';
            end
            
            % endiannes
            if nargin < 3 || isempty(machine)
                machine = 'n';
            end
        
            [fid, msg] = fopen(path, mode, machine);
            
            if fid == -1
                error(sprintf('Could not open file: %s  Error message: %s', ...
                               path, msg));
            end
        end % sfopen
    end % methods
end % Staux

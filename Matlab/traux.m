function out = metin(fun, varargin)
% TRAIN Axilliary functions.
%
% Based on codes by David Bekaert and Andrew Hooper from packages TRAIN
% (https://github.com/dbekaert/TRAIN) and
% StaMPS (https://homepages.see.leeds.ac.uk/~earahoo/stamps/).
%

    switch(fun)
        case 'plot_2d_ftt'
            out = plot_2d_fft(varargin{:});
        case 'butter_filter'
            out = buttter_filter(varargin{:});
        case 'subtract_dry'
            out = subtract_dry(varargin{:});
        case 'invert_abs'
            out = invert_abs(varargin{:});
        case 'invert_phase_stamps'
            out = invert_phase_stamps(varargin{:});
        case 'zero_outlier'
            out = zero_outlier(varargin{:});
        case 'total_aps'
            out = total_aps(varargin{:});
        case 'aps_setup'
            aps_setup(varargin{:})
        case 'geopot2h'
            out = geopot2h(varargin{:})
        case 'calc_wv'
            calc_wv(varargin{:})
        otherwise
            error(['Unknown function ', fun]);
    end
end

function [out] = plot_2d_fft(varargin)
    
    check_matrix = @(x) validateattributes(x, {'numeric'}, ...
                                {'nonempty', 'finite'});
    
    p = inputParser;
    p.FunctionName = 'plot_fft_2d';
    p.addRequired('matrix_fft', check_matrix);
    p.addRequired('out', @ischar);
    p.addParameter('samp_rate', 1, check_matrix);
    p.addParameter('logscale', false, @islogical);
    p.addParameter('fftshift', true, @islogical);
    p.parse(varargin{:});
    
    args = p.Results;
    
    matrix = args.matrix_fft;
    
    nx = size(matrix, 2);
    ny = size(matrix, 1);

    samp_rate = p.Results.samp_rate;
    if isscalar(samp_rate)
        samp_x = 1 / samp_rate;
        samp_y = 1 / samp_rate;
    else
        samp_x = 1 / samp_rate(2);
        samp_y = 1 / samp_rate(1);
    end
    
    if samp_x < 0.0 | samp_y < 0.0
        error('Sampling rate must be positive');
    end
    
    x_fr = (-nx / 2 : nx / 2 - 1) * samp_x / nx;
    y_fr = (-ny / 2 : ny / 2 - 1) * samp_y / ny;

    h = figure('visible', 'off');
    
    if args.fftshift
        matrix = fftshift(matrix);
    end
    
    if args.logscale
        to_plot = log10(abs(matrix));
    else
        to_plot = abs(matrix);
    end
    
    imagesc(x_fr, y_fr, to_plot);
    colorbar();
    saveas(h, args.out);
    
    out.h = h;
    out.x_fr = x_fr;
    out.y_fr = y_fr;
end

function [butter] = butter_filter(varargin)
% Adapted from StaMPS by Andy Hooper

    p = inputParser();
    p.FunctionName = 'butter_filter';
    check_matrix = @(x) validateattributes(x, {'numeric'}, ...
                                {'nonempty', 'positive', 'finite'}, 'fft_2d');
    check_scalar = @(x) validateattributes(x, {'numeric'}, ...
        {'scalar', 'positive', 'finite'}, ...
        'create_butterworth_filter');    
    
    p.addRequired('matrix_size', check_matrix);
    p.addRequired('low_pass_wavelength', check_scalar);
    p.addParameter('order', 5, check_scalar);
    p.addParameter('samp_rate', 1, check_matrix);

    p.parse(varargin{:});
    
    msize = p.Results.matrix_size;
    if isscalar(msize)
        nx = msize;
        ny = msize;
    else
        nx = msize(2);
        ny = msize(1);
    end    
    
    low_pass_wavelength = p.Results.low_pass_wavelength;
    k0 = 1 / low_pass_wavelength;
    
    order = p.Results.order;
    
    samp_rate = p.Results.samp_rate;
    if isscalar(samp_rate)
        samp_x = samp_rate;
        samp_y = samp_rate;
    else
        samp_x = samp_rate(2);
        samp_y = samp_rate(1);
    end
    
    kx = (-nx / 2 : nx / 2 - 1) * samp_x / nx;
    ky = (-ny / 2 : ny / 2 - 1) * samp_y / ny;
        
    [kx, ky] = meshgrid(kx, ky);
    
    k_dist = sqrt( kx.^2 + ky.^2 );
    
    butter = 1 ./ (1 + (k_dist ./ k0).^(2*order));
end

function [abs_phase] = invert_abs(varargin)

    p = inputParser;
    p.FunctionName = 'invert_abs';

    check_phase = @(x) validateattributes(x, {'numeric'}, ...
                                {'nonempty', 'finite', 'ndims', 2});
    check_avg = @(x) validateattributes(x, {'numeric'}, ...
                                {'nonempty', 'vector', 'finite'});
    check_idx = @(x) validateattributes(x, {'numeric'}, ...
                                {'scalar', 'positive', 'finite', 'integer'});
    
    p.addRequired('phase', check_phase);
    p.addParameter('last_row', 0.0, check_avg);
    p.addParameter('master_idx', 1, check_idx);
    
    p.parse(varargin{:});
    
    phase      = p.Results.phase;
    master_idx = p.Results.master_idx;
    last       = p.Results.last_row;
    
    [n_ps, n_ifg] = size(phase);
    
    n_sar = n_ifg + 1;
    
    design = zeros(n_ifg, n_sar);
    design(:, master_idx) = 1.0;
    
    slaves = - diag(ones(n_ifg, 1));
    
    if master_idx == 1
        design(:,2:end) = slaves;
    elseif master_idx == n_sar
        design(:,1:end-1) = slaves;
    else
        idx = master_idx - 1;
        design(1:idx, 1:idx) = slaves(1:idx, 1:idx);
        
        idx = master_idx + 1;
        design(master_idx:end, idx:end) = slaves(master_idx:end, master_idx:end);
    end
    
    clear slaves;
    
    design = [design; ones(1, n_sar)];
    
    if isscalar(last) && last == 0.0
        phase = [phase'; zeros(1, n_ps)];
    else
        if iscolumn(last)
            phase = transpose([phase, last]);
        else
            phase = [transpose(phase); last];
        end
    end
    
    abs_phase = lscov(design, double(phase));
end

function [abs_phase] = invert_phase_stamps(varargin)

    p = inputParser;
    p.FunctionName = 'invert_phase_stamps';

    check_avg = @(x) validateattributes(x, {'numeric'}, ...
                                {'nonempty', 'vector', 'finite'});
    
    p.addParameter('average', 0.0, check_avg);
    
    p.parse(varargin{:});
    
    ph = load('phuw2.mat');
    ps = load('ps2.mat');
    
    phase = ph.ph_uw;
    master_idx = ps.master_ix;
    
    clear ph ps;
 
    phase(:,master_idx) = [];
    
    abs_phase = invert_abs(phase, 'master_idx', master_idx, ...
                           'average', p.Results.average);
end

function [corrected_phase] = subtract_dry()

    ph_uw = load('phuw2.mat');
    ph_uw = ph_uw.ph_uw;
    
    ph_tropo = load('tca2.mat');

    corrected_phase = ph_uw - ph_tropo.ph_tropo_era_hydro;
end

function [d_total] = total_aps(varargin)
% Based on the `aps_weather_model_InSAR` function of the TRAIN packege.

    p = inputParser;
    p.FunctionName = 'total_aps';
    p.addParameter('outdir', '.', @ischar);
    
    p.parse(varargin{:});
    
    outdir = p.Results.outdir;
    
    % radar wavelength in cm
    lambda         = getparm_aps('lambda', 1) * 100;
    ll_matfile     = getparm_aps('ll_matfile',1);
    ifgday_matfile = getparm_aps('ifgday_matfile');
    weather_model_datapath = getparm_aps('era_datapath',1);
    
    % Filename suffix of the output files
    wetoutfile = '_ZWD.xyz';   
    hydroutfile = '_ZHD.xyz'; 
    
    % Filename suffix of the output files
    outfile = '.ztd';
    
    % assumed date structure for era
    datestructure = 'yyyymmdd';
    
    % loading the data
    fprintf('Stamps processed structure \n')
    ps = load(ll_matfile);
    load psver
    dates = ps.day;
    lonlat = ps.lonlat;
    
    %azi_inc = staux('load_azi_inc');
    %inc_angle = azi_inc(:,2);
    
    % getting the dropped ifgs
    drop_ifg_index = getparm('drop_ifg_index');

    % getting the parms file list from stamps to see the final ifg list
    if strcmp(getparm('small_baseline_flag'),'y')
        sb_flag = 1;
    else
        sb_flag = 0;
    end
    
    n_ifg = ps.n_ifg;

    % constructing the matrix with master and slave dates
    if sb_flag ==1
        % for SB
        ifg_number = [1:n_ifg]';
        ifgday_ix = ps.ifgday_ix;
        % removing those dropped interferograms
        ifgday_ix(drop_ifg_index,:) =[];
        ifg_number(drop_ifg_index)=[];

        % defining ix interferograms for which the delay needs to be computed
        ifgs_ix = [ifgday_ix ifg_number];
    else
        % slightly different for PS.
        date_slave_ix = [1:n_ifg]';
        ifg_number = [1:n_ifg]';

        % removing those interferograms that have been dropped
        date_slave_ix(drop_ifg_index)=[];
        ifg_number(drop_ifg_index)=[];

        % the master dates
        date_master_ix = repmat(ps.master_ix,size(date_slave_ix,1),1);

        % ix interferograms
        ifgs_ix = [date_master_ix date_slave_ix ifg_number];
    end
    
    n_dates = length(dates);
    InSAR_datapath = ['.'];
    apsname = fullfile(InSAR_datapath, 'tca', num2str(psver), '.mat');
    apssbname = fullfile(InSAR_datapath, 'tca_sb', num2str(psver), '.mat');

    %% loading the weather model data
    % initialisation
    
    % these are the SAR estimated tropospheric delays for all data
    d_wet   = NaN([size(lonlat,1) n_dates]);
    d_hydro = NaN([size(lonlat,1) n_dates]);
    d_total = NaN([size(lonlat,1) n_dates]);
    flag_wet_hydro_used = 'n';

    ix_no_weather_model_data = [];
    counter = 0;

    % looping over the dates 
    for k = 1:n_dates
        % getting the SAR data and convert it to a string
        date_str = datestr(ps.day(k,1), datestructure);

        % filenames
        model_filename_wet = fullfile(weather_model_datapath, date_str, ...
                                      [date_str, wetoutfile]);
        model_filename_hydro = fullfile(weather_model_datapath, date_str, ...
                                        [date_str, hydroutfile]);
        model_filename = fullfile(weather_model_datapath, date_str, ...
                                  [date_str, outfile]);

        % checking if there is actual data for this date, if not just
        % leave NaN's in the matrix.
        
        if exist(model_filename_wet, 'file') == 2
            flag_wet_hydro_used = 'y';
            
            % computing the dry delay
            [xyz_input, xyz_output] = ...
                    load_weather_model_SAR(model_filename_hydro, double(lonlat));
            
            % saving the data which have not been interpolated
            d_hydro(:,k) = xyz_output(:,3);
            clear xyz_input xyz_output
            
            
            % computing the wet delays
            [xyz_input, xyz_output] = ...
                        load_weather_model_SAR(model_filename_wet, double(lonlat));
             
            % saving the output data
            d_wet(:,k) = xyz_output(:,3);
            clear xyz_output
            counter = counter+1;
        elseif exist(model_filename, 'file') == 2
            flag_wet_hydro_used = 'n';
            
            % this is the GACOS model file, will need to pass the model-type as
            % its grid-note 
            
            [xyz_input, xyz_output] = ...
                load_weather_model_SAR(model_filename, double(lonlat), [], model_type);
            
            % saving the output data
            d_total(:,k) = xyz_output(:,3);
            clear xyz_output
            counter = counter+1;
        else
            % rejected list of weather model images
            ix_no_weather_model_data = [ix_no_weather_model_data k];
        end
        clear model_filename_hydro model_filename_wet date_str
    end
    fprintf([num2str(counter) ' out of ' num2str(n_dates) ...
              ' SAR images have a tropospheric delay estimated \n'])


    %% Computing the type of delay
    if strcmpi(flag_wet_hydro_used,'y')
        d_total = d_hydro + d_wet;
    end

    if ~isnan(outfile)
        save(outfile, 'd_total', 'd_hydro', 'd_wet', '-v7.3');
    end
    
    staux('save_binary', d_total, fullfile(outdir, 'd_total.dat'));
    staux('save_binary', d_hydro, fullfile(outdir, 'd_hydro.dat'));
    staux('save_binary', d_wet,   fullfile(outdir, 'd_wet.dat'));

    %% Converting the Zenith delays to a slant delay
    %if size(inc_angle, 2) > 1 && size(inc_angle, 1) == 1
        %inc_angle = inc_angle';
    %end
    
    %if size(inc_angle, 2) == 1
        %inc_angle = repmat(inc_angle, 1, size(d_total, 2));
        %if size(inc_angle, 1) == 1
            %inc_angle = repmat(inc_angle, size(d_total,1), 1);
        %end
    %end
    
    % d_total = d_total ./ cos(deg2rad(inc_angle));
    %% Converting the range delay to a phase delay
    % converting to phase delay. 
    % The sign convention is such that ph_corrected = ph_original - ph_tropo*
    % d_total = -4 * pi ./ lambda .* d_total;    
end

function [Z] = geopot2h(varargin)

    p = inputParser();
    p.FunctionName = 'geopot2h';
    p.addRequired('geopot', @ismatrix);
    p.addRequired('latgrid', @ismatrix);

    p.parse(varargin{:});
    geopot = p.Results.geopot;
    latgrid = p.Results.latgrid;
    
    % Convert Geopotential to Geopotential Height and then to Geometric Height
    g0 = 9.80665;
    % Calculate Geopotential Height, H
    H = geopot./g0;

    % map of g with latitude
    g = 9.80616.*(1 - 0.002637.*cosd(2.*latgrid) + 0.0000059.*(cosd(2.*latgrid)).^2);
    % map of Re with latitude
    Rmax = 6378137; 
    Rmin = 6356752;
    Re = sqrt(1./(((cosd(latgrid).^2)./Rmax^2) + ((sind(latgrid).^2)./Rmin^2)));

    % Calculate Geometric Height, Z
    Z = (H.*Re)./(g/g0.*Re - H);
end

function [] = aps_setup(varargin)

    p = inputParser();
    p.FunctionName = 'aps_setup';
    p.addRequired('lonlat_step', @isscalar);

    p.parse(varargin{:});
    lonlat_step = p.Results.lonlat_step;
    
    % preprocessor type
    preproc = getparm('insar_processor');

    if strcmp(preproc, 'doris')
        % open dem parameters file
        dem_params = sfopen('demparms.in', 'r');
        dem_path = fgetl(dem_params);
    elseif strcmp(preproc, 'gamma')
        dem_params = '../dem/dem_seg.par';
        dem_path = '../dem/dem_seg_swapped.dem';
    else
        error(['Unrecognized preprocessor ', preproc]);
    end
    
    % get dem file path
    setparm_aps('demfile', dem_path);
    setparm_aps('era_datapath', [pwd '/era']);
    setparm_aps('merra_datapath', [pwd '/merra2']);
    setparm_aps('gacos_datapath', [pwd '/gacos']);
    
    setparm_aps('lambda', getparm('lambda'), 1);
    setparm_aps('heading', getparm('heading'), 1);
    
    % get dem file extension
    [~, ~, ext] = fileparts(dem_path);
    
    % creating rsc file if .dem is the file extension
    if strcmp(ext, '.dem')
        fprintf('DEM extension is .dem, creating, .dem.rsc file\n');
        dem_rsc = sfopen([dem_path, '.rsc'], 'w');
        
        if strcmp(preproc, 'doris')
            % printing dem parameters
            fprintf(dem_rsc, ['WIDTH ', fgetl(dem_params), '\n']);
            fprintf(dem_rsc, ['LENGTH ', fgetl(dem_params), '\n']);
            fprintf(dem_rsc, ['X_FIRST ', fgetl(dem_params), '\n']);        
            fprintf(dem_rsc, ['Y_FIRST ', fgetl(dem_params), '\n']);
        
            % x and y step size
            step = fgetl(dem_params);

            fprintf(dem_rsc, ['X_STEP ', step, '\n']);
            fprintf(dem_rsc, ['Y_STEP ', '-', step, '\n']);
            fprintf(dem_rsc, ['FORMAT ', fgetl(dem_params), '\n']);
            fclose(dem_params);
        elseif strcmp(preproc, 'gamma')
            % printing dem parameters
            fprintf(dem_rsc, ['WIDTH ', num2str(readparm(dem_params, 'width')), '\n']);
            fprintf(dem_rsc, ['LENGTH ', num2str(readparm(dem_params, 'nlines')), '\n']);
            fprintf(dem_rsc, ['X_FIRST ', num2str(readparm(dem_params, 'corner_lon')), '\n']);        
            fprintf(dem_rsc, ['Y_FIRST ', num2str(readparm(dem_params, 'corner_lat')), '\n']);
        
            fprintf(dem_rsc, ['X_STEP ', num2str(readparm(dem_params, 'post_lon')), '\n']);
            fprintf(dem_rsc, ['Y_STEP ', num2str(readparm(dem_params, 'post_lat')), '\n']);
            fprintf(dem_rsc, ['FORMAT ', readparm(dem_params, 'data_format'), '\n']);
        end
        fclose(dem_rsc);
    end
    
    % loading lonlat values, assuming processing was done by StaMPS
    ps2 = load('ps2.mat');
    lonlat = ps2.lonlat;
    clear ps2;
    
    setparm_aps('region_lon', [min(lonlat(:,1)) - lonlat_step, ...
                               max(lonlat(:,1)) + lonlat_step]);

    setparm_aps('region_lat', [min(lonlat(:,2)) - lonlat_step, ...
                               max(lonlat(:,2)) + lonlat_step]);    
end

function [] = calc_wv(varargin)
    
    %p = inputParser;
    lambda = getparm_aps('lambda', 1) * 100;
    
    ph = load('phuw2.mat');
    ps = load('ps2.mat');
    
    phase = ph.ph_uw;
    master_idx = ps.master_ix;
    phase(:,master_idx) = [];
    
    ncols = ps.n_image + 2;
    
    clear ps ph
    
    % loading zenith delays
    d_total = staux('load_binary', 'd_total.dat', ncols);
    % lon., lat. coordinates (first two columns) are not required
    d_total = d_total(:,3:end);
    
    d_hydro = staux('load_binary', 'd_hydro.dat', ncols);
    d_hydro = d_hydro(:,3:end);
    
    d_wet = staux('load_binary', 'd_wet.dat', ncols);
    d_wet = d_wet(:,3:end);
    
    azi_inc = staux('load_binary', 'azi_inc.dat', 2);
    inc_angle = azi_inc(:,2); clear azi_inc
    
    if size(inc_angle, 2) > 1 && size(inc_angle, 1) == 1
        inc_angle = inc_angle';
    end
    
    if size(inc_angle, 2) == 1
        inc_angle = repmat(inc_angle, 1, size(d_total, 2));
        if size(inc_angle, 1) == 1
            inc_angle = repmat(inc_angle, size(d_total, 1), 1);
        end
    end
    
    % Converting the zenith delays to slant delays 
    d_total = d_total ./ cos(deg2rad(inc_angle));
    d_hydro = d_hydro ./ cos(deg2rad(inc_angle));
    % phase = phase .* cos(deg2rad(inc_angle));
    
    %% Converting the range delay to a phase delay
    % The sign convention is such that ph_corrected = ph_original - ph_tropo*
    
    % d_total = -4 * pi ./ lambda .* d_total;
    
    % converting phase to delay
    phase = - (lambda / (4 * pi))  .* phase;

    % calculate average total delay for each SAR acquisition
    % d_avg = mean(d_total, 2);
    d_sum = sum(d_total, 2);
    
    abs_phase = transpose(invert_abs(phase, 'master_idx', master_idx, ...
                                     'last_row', d_sum));
    abs_wet = abs_phase - d_hydro;
    
    h = figure('visible', 'off');
    hist(rms(abs_wet - d_wet));
    %xlabel('Inverted water vapour slant delay [cm]');
    %ylabel('ERA water vapour slant delay [cm]');
    saveas(h, 'dinv_rms_wv.png');

    h = figure('visible', 'off');
    hist(rms(abs_phase - d_total));
    % xlabel('Inverted total slant delay [cm]');
    % ylabel('ERA total slant delay [cm]');
    saveas(h, 'dinv_rms_total.png');
    
    staux('save_binary', abs_phase, 'dinv_total.dat');
    staux('save_binary', abs_wet, 'dinv_wet.dat');
end

function [ret] = rms(data)
    ret = sqrt( mean(data.^2, 2) );
end

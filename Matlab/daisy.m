function out = daisy(fun, varargin)

    switch(fun)
        case 'gmtfiles'
            gmtfiles(varargin{:});
        otherwise
            error(['Unknown function ', fun]);
    end
end

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

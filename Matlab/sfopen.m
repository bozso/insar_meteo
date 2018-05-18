function fid = sfopen(path, mode, machine)

    if nargin < 1 || isempty(path)
       error('Required argument path is not specified');
    end
    
    if nargin < 2 || isempty(mode)
        mode = 'r';
    end
    
    if nargin < 3 || isempty(machine)
        machine = 'n';
    end

    [fid, msg] = fopen(path, mode, machine);
    
    if fid == -1
        error(['Could not open file: ', path, '\nError message: ', msg]);
    end
end
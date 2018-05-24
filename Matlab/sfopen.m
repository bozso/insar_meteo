function [fid] = sfopen(path, mode, machine)
% fid = sfopen(path, mode, machine)
%
% A slightly modified fopen function that exits with error if the file defined
% by path cannot be opened.
% Accepts the same arguments as fopen.
%
    
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
        error(sprintf('Could not open file: %s  Error message: %s', path, msg));
    end
end

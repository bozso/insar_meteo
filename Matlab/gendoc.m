function [] = gendoc()
% Generate html documentation with m2html.
%
    cd ..
    addpath /home/istvan/progs/m2html;
    
    m2html('mfiles', 'Matlab', 'htmldir', 'doc');
end

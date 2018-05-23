g = gmt('init', 'test.ps', 'common', '-R20/30/45/50 -JM180p', 'debug');

gmt('colorbar', g, )

%g = gmt('psbasemap -Ba0.5', g);
%g = gmt('psbasemap -Ba0.5', g);

%gmt('finalize', g);

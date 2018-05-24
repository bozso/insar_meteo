g = gmt('init', 'test.ps', 'common', '-R20/30/45/50 -JM180p', 'debug');

%gmt('colorbar', g, )

g = gmt(g, 'psbasemap -Ba0.5');
g = gmt(g, 'psbasemap -Ba0.5');

gmt('finalize', g);

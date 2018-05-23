g = gmt('init', 'test.ps', '-R20/30/45/50 -JM2.5i');

g = gmt('psbasemap -Ba0.5', g);
g = gmt('psbasemap -Ba0.5', g);

gmt('finalize', g);

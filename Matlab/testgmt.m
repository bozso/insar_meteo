g = gmt('init', 'test.ps');

g = gmt('psbasemap -R20/30/45/50 -JM5i', g);
g = gmt('psbasemap -R20/30/45/50 -JM5i', g);

gmt('finalize', g);

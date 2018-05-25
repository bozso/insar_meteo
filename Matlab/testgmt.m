gmt = Gmt.init('test.ps', 'common', '-JM350p -R20/30/40/50', 'debug');

gmt = Gmt.call('psbasemap -Ba1g1f0.5', gmt);

Gmt.finalize(gmt);

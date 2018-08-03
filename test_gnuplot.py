import numpy as np

def f(self):
    print(self.shape)

f(np.ndarray((2,3)))

#from inmet.gnuplot import Gnuplot, linedef

#gp = Gnuplot(persist=1, debug=1)

##a = gp.grid_data([[1.,2.,3.], [4.,5.,6.]], temp=1, binary=1)
#a = gp.data([1.,2.,3.], [4.,5.,6.], [1., 0.5, 1.2,], using="1:3",
            #vith=linedef(line_type="black", pt_type="empty_circle", pt_size=2.0))

#gp.plot(a)

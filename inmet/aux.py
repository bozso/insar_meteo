from inmet import CLib

inmet = CLib("inmet_aux", fpath)
inmet.test = inmet.wrap("test", [POINTER(Carray)])


def main():
    #_a1 = np.array([1 for ii in range(1000)], dtype=np.float64)
    #_a1 = np.array([1, 2, 3], dtype=np.float64)
    _a1 = np.random.rand(10000)
    print(_a1.dtype, " ", _a1.shape[0])
    #_a2 = np.array([1 for ii in range(140)], dtype=np.float64)
    
    #a1, a2 = npc(_a1), npc(_a2)
    #print(_a1.__array_interface__)
    a1 = npc(_a1)
    #print(a1.data, ptr)
    
    print("Numpy sum: ", _a1.sum())
    inmet.test(a1)
    
    return 0


if __name__ == "__main__":
    main()


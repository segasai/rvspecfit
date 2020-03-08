import multiprocessing as mp
import numpy as np
np.random.seed(1)
sn=300
parallel=True
xs = np.random.randint(0, int(1e9), size=1000)
import accuracy_test
if parallel:
    poo = mp.Pool(24)
    ret2 = [poo.apply_async(accuracy_test.doone, (_, sn)) for _ in xs]
    ret2 = [_.get() for _ in ret2]
else:
    ret2 = [accuracy_test.doone(_,sn) for _ in xs]
v0,v1, err = np.array(ret2).T
dx=v1-v0
print(np.median(dx))
poo.close()
poo.join()

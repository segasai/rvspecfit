import multiprocessing as mp
import numpy as np
import sys

if __name__ == '__main__':
    np.random.seed(1)
    if len(sys.argv) > 1:
        sn = int(sys.argv[1])
    else:
        sn = 300
    nthreads = 24
    nlam = 400
    nit = 1000
    xs = np.random.randint(0, int(1e9), size=nit)
    import accuracy_test

    if nthreads>1:
        poo = mp.Pool(nthreads)
    ret = []
    for i in xs:
        kw = dict(sn=sn, nlam=nlam)
        args = (i,)
        if nthreads>1:
            ret.append(poo.apply_async(accuracy_test.doone,args, kw))
        else:
            ret.append(accuracy_test.doone(*args, **kw))
    if nthreads>1:
        ret = [_.get() for _ in ret]
    v0, v1, err = np.array(ret).T

    ##for i in range(nit):
    #    print (v0[i],v1[i],err[i])
    dx = v1 - v0
    xind = (err < np.median(err))
    print(np.median(dx), np.median(err), np.std(dx))
    print(np.median(dx[xind]), np.median(err[xind]), np.std(dx[xind]))
    poo.close()
    poo.join()

import atpy,numpy as np,scipy.optimize,numpy
import multiprocessing as mp
import matplotlib.pyplot as plt,os,subprocess
from idlsave import idlsave
import sys
import read_grid

def get_revision():
    fname = os.path.dirname(os.path.realpath(__file__))
    tmpout = subprocess.Popen(
        'cd ' + fname + ' ; git log -n 1 --pretty=format:%H -- make_nd.py',
        shell=True, bufsize=80, stdout=subprocess.PIPE).stdout
    revision = tmpout.next()
    return revision
git_rev = ''

def res_fit(p, spec=None, xgrid=None):
	model = (p[0] + p[1] * xgrid)
	res = (spec - model)
	ind = numpy.argsort(res)
	l = len(res)
	frac = 0.2
	l1 = frac * l
	val = ((res[ind[l1:]])**2).sum()
	#print p,val
	return val


def get_cont(lam,spec):
	npix = len(lam)
	npix2= npix//2
	lam1,lam2 = [np.median(_) for _ in [lam[:npix2],lam[npix2:]]]
	sp1,sp2 = [np.median(_) for _ in [spec[:npix2],spec[npix2:]]]
	cont = np.exp(scipy.interpolate.UnivariateSpline([lam1,lam2],
			np.log(np.r_[sp1,sp2]),
			s=0,k=1,ext=0)(lam))
	return cont

def get_fname(HR,postf,curid):
	return 'out/xx_%s%s_%d.nz'%(HR,postf,curid)

class si:
	mat=None
	lamgrid=None

def processer(g, t, m, al, curid):

	lam, spec = read_grid.get_spec(g, t, m, al)
	spec = read_grid.apply_rebinner(si.mat, spec)
	spec1 = spec/get_cont(si.lamgrid,spec)
	spec1 = np.log(spec1) # log the spectrum
	if not numpy.isfinite(spec1).all():
		raise Exception('nans %s'%str((t,g,m,al)))
	spec1=spec1.astype(np.float32)
	print ('x')
	#np.savez(get_fname(HR,postf,curid),spec1)
	return spec1

def doit(setupInfo, postf=''):
	git_rev=''

	tab = atpy.Table('sqlite','/tmp/files.db')           
	ids = (tab.id).astype(int)
	vec = np.array((tab.teff,tab.logg,tab.met,tab.alpha)) 
	#specs=np.zeros((ind.sum(),14139))

	i = 0	

	templ_lam, spec = read_grid.get_spec(4.5, 12000, 0, 0)
	HR,lamleft,lamright,resol,step = setupInfo
	deltav = 1000.
	fac1 = (1 + deltav / 3e5)
	
	lamgrid = np.arange(lamleft / fac1, (lamright + step) * fac1, step)	
	mat = read_grid.make_rebinner_new(templ_lam,lamgrid,resol)

	specs = []
	si.mat= mat
	si.lamgrid= lamgrid
	pool = mp.Pool(8)
	for t,g,m,al in vec.T:
		curid = ids[i]
		i += 1
		print( i)
		specs.append(pool.apply_async(processer,(g,t,m,al,curid)))
	lam  = lamgrid 
	for i in range(len(specs)):
		specs[i]=specs[i].get()	

	specs=np.array(specs)
	idlsave.save('psavs/specs_%s%s.psav'%(HR,postf),'specs,vec,lam',specs,vec,lam)

if __name__=='__main__':
	#doit(('1700B',3600,4500,3500,0.33))
	#doit(('1700D',8300,9000,10000,0.24))
	doit(('GIR_LR9',8100,9400,6560,0.2))
 	
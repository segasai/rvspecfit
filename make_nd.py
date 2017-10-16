from idlsave import idlsave
import numpy as np,numpy.random
import scipy.interpolate.interpnd,os,subprocess
from random import random,seed
import pickle
import dill

def get_revision():
	""" get the git revision of the code"""
	try:
		fname = os.path.dirname(os.path.realpath(__file__))
		tmpout = subprocess.Popen(
	   'cd ' + fname + ' ; git log -n 1 --pretty=format:%H -- make_nd.py',
	   shell=True, bufsize=80, stdout=subprocess.PIPE).stdout
		revision = tmpout.read()
		return revision
	except:
		return ''

git_rev = get_revision()

hrs = ['GIR_LR9']

def mapper(v):
	import numpy as np
	return np.array([np.log10(v[0]),v[1],v[2],v[3]])

def invmapper(v):
	import numpy as np
	return np.array( [10**v[0],v[1],v[2],v[3]] )

postf=''
for i_HR, HR in enumerate(hrs):
	vec, specs,lam = idlsave.restore('psavs/specs_%s%s.psav'%(HR, postf),'vec,specs,lam')
	vec = vec.astype(float)
	vec = mapper(vec)
	ndim = 4 
	delta = 1
	lspans = vec.min(axis=1) - delta
	rspans = vec.max(axis=1) + delta	
	positions=[]
	seed(1)
	perturbation = 1e-6
	for i in range(2**ndim):
		curpos=[]
		for j in range(ndim):
			flag = (i&(2**j))>0
			curspan = [lspans[j],rspans[j]]
			curpos.append(curspan[flag])
		positions.append(curpos)
		# add fake points, to detect out of the grid poinds
	vec0 = vec.copy()
	vec = np.hstack((vec,np.array(positions).T))
	vec = vec + np.random.uniform(-perturbation,perturbation,size=vec.shape)

	nspec,lenspec= specs.shape
	fakespec = np.ones(lenspec)

	specs = np.append(specs, np.tile(fakespec,(2**ndim,1)),axis=0)
	extraflags = np.concatenate((np.zeros(nspec),np.ones(2**ndim)))
	vec = vec.astype(np.float64)
	extraflags = extraflags.astype(np.float64)
	specs = specs.astype(np.float64)
	
	extraflags = extraflags[:, None]
	triang = scipy.spatial.Delaunay(vec.T)
	
	savefile = 'psavs/interp_%s%s.pkl'%(HR, postf)
	dHash = {}
	dHash['lam'] = lam
	dHash['triang'] = triang
	dHash['extraflags'] = extraflags
	dHash['vec'] = vec
	dHash['mapper'] = dill.dumps(mapper)
	dHash['invmapper'] = dill.dumps(invmapper)
	with open(savefile,'wb') as fp:
		pickle.dump(dHash, fp)
	np.save('psavs/interpdat_%s%s.npy'%(HR, postf), np.asfortranarray(specs))

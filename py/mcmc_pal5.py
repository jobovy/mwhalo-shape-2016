###############################################################################
# mcmc_pal5.py: module to run MCMC analysis of the Pal 5 stream
###############################################################################
import sys
import os, os.path
import copy
import time
import pickle
import csv
from optparse import OptionParser
import subprocess
import warnings
import numpy
from scipy.misc import logsumexp
import emcee
import pal5_util
_DATADIR= os.getenv('DATADIR')
def get_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    # Potential parameters
    parser.add_option("--bf_b15",action="store_true", 
                      dest="bf_b15",default=False,
                      help="If set, use the best-fit to the MWPotential2014 data")
    parser.add_option("--seed",dest='seed',default=1,type='int',
                      help="seed for everything except for potential")
    parser.add_option("--fitsigma",action="store_true", 
                      dest="fitsigma",default=False,
                      help="If set, fit for the velocity-dispersion parameter")
    parser.add_option("--dt",dest='dt',default=10.,type='float',
                      help="Run MCMC for this many minutes")
    parser.add_option("-i",dest='pindx',default=None,type='int',
                      help="Index into the potential samples to consider")
    parser.add_option("--ro",dest='ro',default=pal5_util._REFR0,type='float',
                      help="Distance to the Galactic center in kpc")
    parser.add_option("--td",dest='td',default=5.,type='float',
                      help="Age of the stream")
    parser.add_option("--samples_savefilename",
                      dest='samples_savefilename',
                      default='mwpot14varyc-samples.pkl',
                      help="Name of the file that contains the potential samples")
    # Output file
    parser.add_option("-o",dest='outfilename',
                      default=None,
                      help="Name of the file that will hold the output")
    # Multi-processing
    parser.add_option("-m",dest='multi',default=1,type='int',
                      help="Number of CPUs to use for streamdf setup")
    return parser


def filelen(filename):
    p= subprocess.Popen(['wc','-l',filename],stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
    result,err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def load_samples(options):
    if os.path.exists(options.samples_savefilename):
        with open(options.samples_savefilename,'rb') as savefile:
            s= pickle.load(savefile)
    else:
        raise IOError("File %s that is supposed to hold the potential samples does not exist" % options.samples_savefilename)
    return s

def find_starting_point(options,pot_params,dist,pmra,pmdec,sigv):
    # Find a decent starting point, useTM to speed this up, bc it doesn't matter much
    interpcs=[0.65,0.75,0.875,1.,1.125,1.25,1.5,1.65]
    cs= numpy.arange(0.7,1.61,0.01)
    pal5varyc_like= pal5_util.predict_pal5obs(pot_params,cs,
                                              dist=dist,pmra=pmra,pmdec=pmdec,
                                              sigv=sigv,td=options.td,
                                              ro=options.ro,vo=220.,
                                              interpk=1,
                                              interpcs=interpcs,useTM=True,
                                              trailing_only=True,verbose=False)
    pos_radec, rvel_ra= pal5_util.pal5_data()
    if options.fitsigma:
        lnlike= numpy.sum(\
            pal5_util.pal5_lnlike(pos_radec,rvel_ra,
                                  pal5varyc_like[0],
                                  pal5varyc_like[1],
                                  pal5varyc_like[2],
                                  pal5varyc_like[3],
                                  pal5varyc_like[4],
                                  pal5varyc_like[5],
                                  pal5varyc_like[6])[:,:3:2],axis=1)
    else:
        # For each one, move the track up and down a little to simulate sig changes
        deco= numpy.linspace(-0.5,0.5,101)
        lnlikes= numpy.zeros((len(cs),len(deco)))-100000000000000000.
        for jj,do in enumerate(deco):
            tra= pal5varyc_like[0]
            tra[:,:,1]+= do
            lnlikes[:,jj]= pal5_util.pal5_lnlike(pos_radec,rvel_ra,
                                                 tra,pal5varyc_like[1],
                                                 pal5varyc_like[2],
                                                 pal5varyc_like[3],
                                                 pal5varyc_like[4],
                                                 pal5varyc_like[5],
                                                 pal5varyc_like[6])[:,0]
        lnlike= numpy.amax(lnlikes,axis=1)
    return cs[numpy.argmax(lnlike)]

def lnp(p,pot_params,options):
    warnings.filterwarnings("ignore",
                            message="Using C implementation to integrate orbits")
    #p=[c,vo/220,dist/22.,pmo_parallel,pmo_perp] and ln(sigv) if fitsigma
    c= p[0]
    vo= p[1]*pal5_util._REFV0
    dist= p[2]*22.
    pmra= -2.296+p[3]+p[4]
    pmdecpar= 2.257/2.296
    pmdecperp= -2.296/2.257
    pmdec= -2.257+p[3]*pmdecpar+p[4]*pmdecperp
    if options.fitsigma:
        sigv= 0.4*numpy.exp(p[5])
    else:
        sigv= 0.4
    # Priors
    if c < 0.5: return -100000000000000000.
    elif c > 2.: return -10000000000000000.
    elif vo < 200: return -10000000000000000.
    elif vo > 250: return -10000000000000000.
    elif dist < 19.: return -10000000000000000.
    elif dist > 24.: return -10000000000000000.
    elif options.fitsigma and sigv < 0.1: return -10000000000000000.
    elif options.fitsigma and sigv > 1.: return -10000000000000000.
    # Setup the model
    pal5varyc_like= pal5_util.predict_pal5obs(pot_params,c,singlec=True,
                                              dist=dist,pmra=pmra,pmdec=pmdec,
                                              ro=options.ro,vo=vo,
                                              trailing_only=True,verbose=False,
                                              sigv=sigv,td=options.td,
                                              useTM=False,
                                              nTrackChunks=8)
    pos_radec, rvel_ra= pal5_util.pal5_data()
    if options.fitsigma:
        lnlike= pal5_util.pal5_lnlike(pos_radec,rvel_ra,
                                      pal5varyc_like[0],
                                      pal5varyc_like[1],
                                      pal5varyc_like[2],
                                      pal5varyc_like[3],
                                      pal5varyc_like[4],
                                      pal5varyc_like[5],
                                      pal5varyc_like[6])
        if not pal5varyc_like[7]: addllnlike= -15. # penalize 
        else: addllnlike= 0.
        #print addllnlike, pal5varyc_like[7]
        #sys.stdout.flush()
        return lnlike[0,0]+lnlike[0,2]+addllnlike+\
            -0.5*(pmra+2.296)**2./0.186**2.-0.5*(pmdec+2.257)**2./0.181**2.
    # If not fitsigma, move the track up and down a little to simulate sig changes
    deco= numpy.linspace(-0.5,0.5,101)
    lnlikes= numpy.zeros(len(deco))-100000000000000000.
    for jj,do in enumerate(deco):
        tra= copy.deepcopy(pal5varyc_like[0])
        tra[:,:,1]+= do
        lnlikes[jj]= pal5_util.pal5_lnlike(pos_radec,rvel_ra,
                                           tra,pal5varyc_like[1],
                                           pal5varyc_like[2],
                                           pal5varyc_like[3],
                                           pal5varyc_like[4],
                                           pal5varyc_like[5],
                                           pal5varyc_like[6])[0,0]
    return logsumexp(lnlikes)\
        +pal5_util.pal5_lnlike(pos_radec,rvel_ra,
                               pal5varyc_like[0],pal5varyc_like[1],
                               pal5varyc_like[2],pal5varyc_like[3],
                               pal5varyc_like[4],pal5varyc_like[5],
                               pal5varyc_like[6])[0,2]\
        -0.5*(pmra+2.296)**2./0.186**2.-0.5*(pmdec+2.257)**2./0.181**2.

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    # Set random seed for potential selection
    numpy.random.seed(1)
    # Load potential parameters
    if options.bf_b15:
        pot_params= [0.60122692,0.36273147,-0.97591502,-3.34169377,
                     0.71877924,-0.01519337,-0.01928001]
    else:
        pot_samples= load_samples(options)
        rndindx= numpy.random.permutation(pot_samples.shape[1])[options.pindx]
        pot_params= pot_samples[:,rndindx]
    print pot_params
    # Now set the seed for the MCMC
    numpy.random.seed(options.seed)
    nwalkers= 10+2*options.fitsigma
    # For a fiducial set of parameters, find a good fit to use as the starting 
    # point
    all_start_params= numpy.zeros((nwalkers,5+options.fitsigma))
    start_lnprob0= numpy.zeros(nwalkers)
    if not os.path.exists(options.outfilename):
        pmra= -2.296
        pmdec= -2.257
        dist= 23.2
        #cstart= find_starting_point(options,pot_params,dist,pmra,pmdec,0.4)
        cstart= 1.
        if cstart > 1.15: cstart= 1.15 # Higher c doesn't typically really work
        if options.fitsigma:
            start_params= numpy.array([cstart,1.,dist/22.,0.,0.,0.])
            step= numpy.array([0.05,0.05,0.05,0.05,0.01,0.05])
        else:
            start_params= numpy.array([cstart,1.,dist/22.,0.,0.])
            step= numpy.array([0.05,0.05,0.05,0.05,0.01])
        nn= 0
        while nn < nwalkers:
            all_start_params[nn]= start_params\
                +numpy.random.normal(size=len(start_params))*step
            start_lnprob0[nn]= lnp(all_start_params[nn],pot_params,options)
            if start_lnprob0[nn] > -1000000.: 
                print all_start_params[nn], start_lnprob0[nn]
            if start_lnprob0[nn] > -1000000.: nn+= 1
    else:
        # Get the starting point from the output file
        with open(options.outfilename,'rb') as savefile:
            all_lines= savefile.readlines()
        for nn in range(nwalkers):
            lastline= all_lines[-1-nn]
            tstart_params= numpy.array([float(s) for s in lastline.split(',')])
            start_lnprob0[nn]= tstart_params[-1]
            all_start_params[nn]= tstart_params[:-1]
    # Output
    if os.path.exists(options.outfilename):
        outfile= open(options.outfilename,'a',0)
    else:
        # Setup the file
        outfile= open(options.outfilename,'w',0)
        outfile.write('# potparams:%.8f,%.8f,%.8f,%.8f,%.8f\n' % \
                          (pot_params[0],pot_params[1],pot_params[2],
                           pot_params[3],pot_params[4]))
        for nn in range(nwalkers):
            if options.fitsigma:
                outfile.write('%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n' % \
                                  (all_start_params[nn,0],all_start_params[nn,1],
                                   all_start_params[nn,2],all_start_params[nn,3],
                                   all_start_params[nn,4],all_start_params[nn,5],
                                   start_lnprob0[nn]))
            else:
                outfile.write('%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n' % \
                                  (all_start_params[nn,0],all_start_params[nn,1],
                                   all_start_params[nn,2],all_start_params[nn,3],
                                   all_start_params[nn,4],start_lnprob0[nn]))
        outfile.flush()
    outwriter= csv.writer(outfile,delimiter=',')
    # Run MCMC
    sampler= emcee.EnsembleSampler(nwalkers,all_start_params.shape[1],
                                   lnp,args=(pot_params,options),
                                   threads=options.multi)
    rstate0= numpy.random.mtrand.RandomState().get_state()
    start= time.time()
    while time.time() < start+options.dt*60.:
        new_params, new_lnp, new_rstate0=\
            sampler.run_mcmc(all_start_params,1,lnprob0=start_lnprob0,
                             rstate0=rstate0,storechain=False)
        all_start_params= new_params
        start_lnprob0= new_lnp
        rstate0= new_rstate0
        for nn in range(nwalkers):
            if options.fitsigma:
                outfile.write('%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n' % \
                                  (all_start_params[nn,0],all_start_params[nn,1],
                                   all_start_params[nn,2],all_start_params[nn,3],
                                   all_start_params[nn,4],all_start_params[nn,5],
                                   start_lnprob0[nn]))
            else:
                outfile.write('%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n' % \
                                  (all_start_params[nn,0],all_start_params[nn,1],
                                   all_start_params[nn,2],all_start_params[nn,3],
                                   all_start_params[nn,4],start_lnprob0[nn]))
        outfile.flush()
    outfile.close()            
    

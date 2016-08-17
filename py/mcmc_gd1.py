###############################################################################
# mcmc_gd1.py: module to run MCMC analysis of the GD-1 stream
###############################################################################
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
import gd1_util
_DATADIR= os.getenv('DATADIR')
def get_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    # Potential parameters
    parser.add_option("--bf_b15",action="store_true", 
                      dest="bf_b15",default=False,
                      help="If set, use the best-fit to the MWPotential2014 data")
    parser.add_option("--logpot",action="store_true", 
                      dest="logpot",default=False,
                      help="If set, use a logarithmic flattened potential")
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

def lnp(p,pot_params,options):
    warnings.filterwarnings("ignore",
                            message="Using C implementation to integrate orbits")
    #p=[c,vo/220,phi2,dist/10.,pmphi1+8.5,pmphi2+2,vlos/300] and ln(sigv) if fitsigma
    c= p[0]
    vo= p[1]*pal5_util._REFV0
    phi2= p[2]
    dist= p[3]*10.
    pmphi1= -8.5+p[4]
    pmphi2= -2.0+p[5]
    vlos= p[6]*300.
    if options.fitsigma:
        sigv= 0.4*numpy.exp(p[7])
    else:
        sigv= 0.4
    # Priors
    if c < 0.5: return -100000000000000000.
    elif c > 2.: return -10000000000000000.
    elif not options.logpot and vo < 200: return -10000000000000000.
    elif not options.logpot and vo > 250: return -10000000000000000.
    elif options.logpot and vo < 130.: return -10000000000000000.
    elif options.logpot and vo > 290.: return -10000000000000000.
    elif dist < 5.: return -10000000000000000.
    elif dist > 15.: return -10000000000000000.
    elif options.fitsigma and sigv < 0.1: return -10000000000000000.
    elif options.fitsigma and sigv > 1.: return -10000000000000000.
    # Setup the model
    gd1varyc_like= gd1_util.predict_gd1obs(pot_params,c,
                                           phi1=0.,
                                           phi2=phi2,
                                           dist=dist,
                                           pmphi1=pmphi1,
                                           pmphi2=pmphi2,
                                           vlos=vlos,
                                           ro=options.ro,vo=vo,
                                           verbose=False,
                                           sigv=sigv,td=options.td,
                                           useTM=False,
                                           nTrackChunks=8,
                                           logpot=options.logpot)
    posdata,distdata,pmdata,rvdata= gd1_util.gd1_data()
    lnlike= gd1_util.gd1_lnlike(posdata,distdata,pmdata,rvdata,
                                gd1varyc_like[0])
    if not gd1varyc_like[1]: addllnlike= -15. # penalize 
    else: addllnlike= 0.
    out= numpy.sum(lnlike)+addllnlike
    if numpy.isnan(out): return -10000000000000000.
    else: return out

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    # Set random seed for potential selection
    numpy.random.seed(1)
    # Load potential parameters
    if options.bf_b15:
        pot_params= [0.60122692,0.36273147,-0.97591502,-3.34169377,
                     0.71877924,-0.01519337,-0.01928001]
    elif options.logpot:
        pot_params= [-1,-1,-1,-1,-1]
    else:
        pot_samples= load_samples(options)
        rndindx= numpy.random.permutation(pot_samples.shape[1])[options.pindx]
        pot_params= pot_samples[:,rndindx]
    print pot_params
    # Now set the seed for the MCMC
    numpy.random.seed(options.seed)
    nwalkers= 14+2*options.fitsigma
    # Compute the likelihoods of a first set
    all_start_params= numpy.zeros((nwalkers,7+options.fitsigma))
    start_lnprob0= numpy.zeros(nwalkers)
    if not os.path.exists(options.outfilename):
        cstart= 1.
        if options.fitsigma:
            start_params= numpy.array([cstart,1.,-1.,
                                       10.2/10.,0.,-0.05,-285./300.,
                                       numpy.log(0.365/0.4)])
            step= numpy.array([0.05,0.03,0.05,0.03,0.05,0.03,0.05,0.05])
        else:
            start_params= numpy.array([cstart,1.,-1.,
                                       10.2/10.,0.,-0.05,-285./300.])
            step= numpy.array([0.05,0.03,0.05,0.03,0.05,0.03,0.05])
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
                outfile.write('%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n' % \
                                  (all_start_params[nn,0],all_start_params[nn,1],
                                   all_start_params[nn,2],all_start_params[nn,3],
                                   all_start_params[nn,4],all_start_params[nn,5],
                                   all_start_params[nn,6],all_start_params[nn,7],
                                   start_lnprob0[nn]))
            else:
                outfile.write('%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n' % \
                                  (all_start_params[nn,0],all_start_params[nn,1],
                                   all_start_params[nn,2],all_start_params[nn,3],
                                   all_start_params[nn,4],all_start_params[nn,5],
                                   all_start_params[nn,6],start_lnprob0[nn]))
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
                outfile.write('%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n' % \
                                  (all_start_params[nn,0],all_start_params[nn,1],
                                   all_start_params[nn,2],all_start_params[nn,3],
                                   all_start_params[nn,4],all_start_params[nn,5],
                                   all_start_params[nn,6],all_start_params[nn,7],
                                   start_lnprob0[nn]))
            else:
                outfile.write('%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n' % \
                                  (all_start_params[nn,0],all_start_params[nn,1],
                                   all_start_params[nn,2],all_start_params[nn,3],
                                   all_start_params[nn,4],all_start_params[nn,5],
                                   all_start_params[nn,6],start_lnprob0[nn]))
        outfile.flush()
    outfile.close()            
    

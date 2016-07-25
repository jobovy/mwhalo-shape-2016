###############################################################################
# analyze_pal5.py: module to run analysis of the Pal 5 stream re: halo shape
###############################################################################
import os, os.path
import pickle
import csv
from optparse import OptionParser
import subprocess
import numpy
import pal5_util
import MWPotential2014Likelihood
_DATADIR= os.getenv('DATADIR')
def get_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    # Potential parameters
    parser.add_option("--bf_b15",action="store_true", 
                      dest="bf_b15",default=False,
                      help="If set, use the best-fit to the MWPotential2014 data")
    parser.add_option("--seed",dest='seed',default=1,type='int',
                      help="seed for potential parameter selection and everything else")
    parser.add_option("-i",dest='pindx',default=None,type='int',
                      help="Index into the potential samples to consider")
    parser.add_option("--ro",dest='ro',default=pal5_util._REFR0,type='float',
                      help="Distance to the Galactic center in kpc")
    parser.add_option("--vo",dest='vo',default=pal5_util._REFV0,type='float',
                      help="Circular velocity at ro in km/s")
    parser.add_option("--samples_savefilename",
                      dest='samples_savefilename',
                      default='mwpot14varyc-samples.pkl',
                      help="Name of the file that contains the potential samples")
    # c grid
    parser.add_option("--cmin",dest='cmin',default=0.5,type='float',
                      help="Minimum c to consider")
    parser.add_option("--cmax",dest='cmax',default=1.5,type='float',
                      help="Maximum c to consider")
    parser.add_option("--cstep",dest='cstep',default=0.01,type='float',
                      help="C resolution")
    # Heuristic guess of the maximum
    parser.add_option("--alongbfpm",action="store_true", 
                      dest="alongbfpm",default=False,
                      help="If set, move along the best-fit distance in the heuristic guess")
    # Distances grid
    parser.add_option("--dmin",dest='dmin',default=21.,type='float',
                      help="Minimum distance to consider in guess")
    parser.add_option("--dmax",dest='dmax',default=25.,type='float',
                      help="Maximum distance to consider in guess")
    parser.add_option("--dstep",dest='dstep',default=0.02,type='float',
                      help="Distance resolution")
    parser.add_option("--dgridmin",dest='dgridmin',default=-6,type='int',
                      help="Minimum distance to consider in the final grid, in units of dstep from guess")
    parser.add_option("--dgridmax",dest='dgridmax',default=7,type='int',
                      help="Maximum distance to consider in the final grid, in units of dstep from guess")
    # PM offsets grid
    parser.add_option("--pmmin",dest='pmmin',default=-0.02,type='float',
                      help="Minimum proper motion offset to consider in guess")
    parser.add_option("--pmmax",dest='pmmax',default=0.02,type='float',
                      help="Maximum proper motion offset to consider in guess")
    parser.add_option("--pmstep",dest='pmstep',default=0.001,type='float',
                      help="Proper-motion offset resolution")
    parser.add_option("--pmgridmin",dest='pmgridmin',default=-3,type='int',
                      help="Minimum proper motion offset to consider in the final grid, in units of pmstep from guess")
    parser.add_option("--pmgridmax",dest='pmgridmax',default=4,type='int',
                      help="Maximum proper motion offset to consider in the final grid, in units of pmstep from guess")
    # Multi-processing
    parser.add_option("-m",dest='multi',default=8,type='int',
                      help="Number of CPUs to use for streamdf setup")
    # Increase the uncertainties?
    parser.add_option("--emult",dest='emult',default=1.,type='float',
                      help="If set, multiply the uncertainties by this factor")
    # Output file
    parser.add_option("-o",dest='outfilename',
                      default=None,
                      help="Name of the file that will hold the output")
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

def analyze_one_model(options,cs,pot_params,dist,pmra,pmdec):
    # First just compute the interpolation points, to adjust the width and length
    interpcs=[0.5,0.75,0.875,1.,1.125,1.25,1.5,1.75]#,2.,2.25]
    pal5varyc= pal5_util.predict_pal5obs(pot_params,interpcs,
                                         dist=dist,pmra=pmra,pmdec=pmdec,
                                         ro=options.ro,vo=options.vo,
                                         interpk=1,
                                         multi=options.multi,interpcs=interpcs)
    if len(pal5varyc[6]) == 0:
        out= numpy.zeros((len(cs),5))-1000000000000000.
        return out
    sigv=0.4*18./pal5varyc[4]
    td=5.*25./pal5varyc[5]/(sigv/0.4)
    td[td > 14.]= 14. # don't allow older than 14 Gyr
    pal5varyc_like= pal5_util.predict_pal5obs(pot_params,cs,
                                              dist=dist,pmra=pmra,pmdec=pmdec,
                                              ro=options.ro,vo=options.vo,
                                              multi=options.multi,
                                              interpk=1,
                                              interpcs=interpcs,
                                              sigv=sigv,td=td)
    pos_radec, rvel_ra= pal5_util.pal5_data()
    pos_radec[:,2]*= options.emult
    rvel_ra[:,2]*= options.emult
    return pal5_util.pal5_lnlike(pos_radec,rvel_ra,*pal5varyc_like)

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    # Set random seed
    numpy.random.seed(options.seed)
    # Load potential parameters
    if options.bf_b15:
        pot_params= [0.60122692,0.36273147,-0.97591502,-3.34169377,
                     0.71877924,-0.01519337,-0.01928001]
    else:
        pot_samples= load_samples(options)
        rndindx= numpy.random.permutation(pot_samples.shape[1])[options.pindx]
        pot_params= pot_samples[:,rndindx]
    print pot_params
    if os.path.exists(options.outfilename):
        # Read the distance/proper motion grid from the file
        with open(options.outfilename,'rb') as savefile:
            line1= savefile.readline()
            cline= savefile.readline()
            dline= savefile.readline()
            dsline= savefile.readline()
            pmsline= savefile.readline()
        ds= numpy.array([float(d) for d in dsline[1:].split(',')])
        pms= numpy.array([float(pm) for pm in pmsline[1:].split(',')])
    else:
        # Find good distances and proper motions
        pot= MWPotential2014Likelihood.setup_potential(pot_params,1.,
                                                       False,False,
                                                       options.ro,options.vo)
        dpmguess= pal5_util.pal5_dpmguess(pot,alongbfpm=options.alongbfpm,
                                          pmmin=options.pmmin,
                                          pmmax=options.pmmax,
                                          pmstep=options.pmstep,
                                          dmin=options.dmin,dmax=options.dmax,
                                          vo=options.vo,ro=options.ro)
        bestd= dpmguess[0]
        bestpm= dpmguess[1]
        print "Good distance and proper motion are: %.2f, %.2f" \
            % (bestd,bestpm)
        # March along the maximum line if alongbfpm
        ds= []
        pms= []
        for ii in range(options.dgridmin,options.dgridmax):
            for jj in range(options.pmgridmin,options.pmgridmax):
                ds.append(bestd+ii*options.dstep)
                pms.append(bestpm+(ds[-1]-bestd)*0.099+jj*options.pmstep)
    pmdecpar= 2.257/2.296
    # Setup c grid
    cs= numpy.arange(options.cmin,options.cmax+options.cstep/2.,options.cstep)
    if cs[-1] > options.cmax+options.cstep/2.: cs= cs[:-1]
    # Output
    if os.path.exists(options.outfilename):
        # Figure out how many ds were already run from the length of the file
        flen= filelen(options.outfilename)
        start_lines= 5
        line_per_dist= 5
        ii= (flen-start_lines)//line_per_dist
        outfile= open(options.outfilename,'a')
    else:
        # Setup the file
        outfile= open(options.outfilename,'w')
        outfile.write('# potparams:%.8f,%.8f,%.8f,%.8f,%.8f\n' % \
                          (pot_params[0],pot_params[1],pot_params[2],
                           pot_params[3],pot_params[4]))
        outfile.write('# cmin cmax cstep: %.3f,%.3f,%.3f\n' % \
                          (options.cmin,options.cmax,options.cstep))
        outfile.write('# dmin dmax dstep: %.3f,%.3f,%.3f\n' % \
                          (numpy.amin(ds),numpy.amax(ds),options.dstep))
        nextline= '#'
        for ii,d in enumerate(ds):
            nextline+= '%.4f' % d
            if not ii == len(ds)-1: nextline+= ','
        outfile.write(nextline+'\n')
        nextline= '#'
        for ii,pm in enumerate(pms):
            nextline+= '%.4f' % pm
            if not ii == len(pms)-1: nextline+= ','
        outfile.write(nextline+'\n')
        outfile.flush()
        ii= 0
    outwriter= csv.writer(outfile,delimiter=',')
    # Analyze each distance, pmra, pmdec
    while ii < len(ds):
        dist= ds[ii]
        pmo= pms[ii]
        pmra= -2.296+pmo
        pmdec= -2.257+pmo*pmdecpar
        print "Working on %i: dist %.2f, pmra %.3f, pmdec %.3f" \
            % (ii,dist,pmra,pmdec)
        likes= analyze_one_model(options,cs,pot_params,dist,pmra,pmdec).T
        # Write
        for row in likes:
            outwriter.writerow(row)
        outfile.flush()
        ii+= 1
    outfile.close()            
    

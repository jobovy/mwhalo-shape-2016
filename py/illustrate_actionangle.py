# Make a little movie to illustrate how to compute actions
# Usage:
# python illustrate_actionangle.py ../movies/aaI ../actionIllustration.mpg
import sys
import os, os.path
import numpy
import tqdm
import subprocess
import astropy.units as u
from galpy.potential import IsochronePotential
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleIsochrone
from galpy.util import bovy_plot
from matplotlib import pyplot, gridspec
def create_frames(basefilename):
    #if os.path.exists(basefilename+'_00000.png'): return None
    pot= IsochronePotential(normalize=1.,b=1.2)
    aAT= actionAngleIsochrone(ip=pot)
    bmock= 0.8
    aAF= actionAngleIsochrone(b=bmock)
    ro, vo= 8., 220.
    o= Orbit([1.,0.3,0.9,0.2,1.,0.1])
    tfac, skip= 10, 1
    ndesired= 30*25
    times= numpy.linspace(0.,7.,tfac*ndesired)*u.Gyr
    # Subsample
    otimesIndices= (numpy.arange(len(times))/float(len(times)-1))**10\
        *(len(times)-2)
    otimesIndices= numpy.unique(numpy.floor(otimesIndices).astype('int'))
    if len(otimesIndices) > ndesired:
        sfac= int(numpy.floor(len(otimesIndices)/float(ndesired)))
        otimesIndices= otimesIndices[::sfac]
    otimes= times[otimesIndices]
    o.integrate(times,pot)
    # Compute all actions in the wrong potential
    acfs= aAF.actionsFreqsAngles(o.R(times),
                                 o.vR(times),
                                 o.vT(times),
                                 o.z(times),
                                 o.vz(times),
                                 o.phi(times))
    js= (acfs[0],acfs[1],acfs[2])
    danglerI= ((numpy.roll(acfs[6],-1)-acfs[6]) % (2.*numpy.pi))[:-1]
    jrC= numpy.cumsum(acfs[0][:-1]*danglerI)\
        /numpy.cumsum(danglerI)
    # And also the actions in the true potential
    jsT= aAT(o.R(times),o.vR(times),o.vT(times),o.z(times),o.vz(times))
    jrT= numpy.median(jsT[0])*8.*220.
    # Setup gridspec
    gs= gridspec.GridSpec(1,3,wspace=0.325,bottom=0.2,left=0.075)
    # For each time step, plot: orbit, Jr, <Jr>
    for ii in tqdm.tqdm(range(len(otimes))):
        bovy_plot.bovy_print(fig_width=11.2,fig_height=4.,
                             axes_labelsize=17.,text_fontsize=12.,
                             xtick_labelsize=13.,ytick_labelsize=13.)
        pyplot.figure()
        pyplot.subplot(gs[0])
        minIndx= otimesIndices[ii]-100
        if minIndx < 0: minIndx= 0
        bovy_plot.bovy_plot([o.x(otimes[ii])*ro],[o.z(otimes[ii])*ro],
                            'o',ms=15.,gcf=True,
                            xrange=[-19.,19.],
                            yrange=[-19.,19.],
                            xlabel=r'$X\,(\mathrm{kpc})$',
                            ylabel=r'$z\,(\mathrm{kpc})$')
        if ii > 0:
            bovy_plot.bovy_plot(o.x(times[minIndx:otimesIndices[ii]:skip])*ro,
                                o.z(times[minIndx:otimesIndices[ii]:skip])*ro,
                                '-',overplot=True)
        pyplot.subplot(gs[1])
        bovy_plot.bovy_plot([times[otimesIndices[ii]].value],
                            [js[0][otimesIndices[ii]]*ro*vo],
                            'o',ms=15.,gcf=True,
                            xrange=[0.,1.+times[otimesIndices[ii]].value],
                            yrange=[0.,349.],
                            xlabel=r'$\mathrm{time}\,(\mathrm{Gyr})$',
                            ylabel=r'$J_R\,(\mathrm{kpc\,km\,s}^{-1})$')
        if ii > 0:
            bovy_plot.bovy_plot(times[:otimesIndices[ii]:skip].value,
                                js[0][:otimesIndices[ii]:skip]*ro*vo,
                                '-',overplot=True)
        pyplot.axhline(jrT,ls='--',color='k')
        pyplot.subplot(gs[2])
        bovy_plot.bovy_plot([otimes[ii].value],
                            [jrC[otimesIndices[ii]]*ro*vo],
                            'o',ms=15.,gcf=True,
                            xrange=[0.,1.+times[otimesIndices[ii]].value],
                            yrange=[(otimes[ii]/times[-1])**0.3*(jrT-3),
                                    349.\
                                        +(otimes[ii]/times[-1])**0.3\
                                        *(jrT+3-349.)],
                            xlabel=r'$\mathrm{time}\,(\mathrm{Gyr})$',
                            ylabel=r'$J_R\,(\mathrm{kpc\,km\,s}^{-1})$')
        if ii > 0:
            bovy_plot.bovy_plot(times[:otimesIndices[ii]:skip].value,
                                jrC[:otimesIndices[ii]:skip]*ro*vo,
                                '-',overplot=True)
        pyplot.axhline(jrT,ls='--',color='k')       
        bovy_plot.bovy_end_print(basefilename+'_%05d.png' % ii)
    return None
def create_movie(basefilename,outputfilename):
    framerate= 25
    bitrate= 1000000
    try:
        subprocess.check_call(['ffmpeg',
                               '-i',
                               basefilename+'_%05d.png',
                               '-y',
                               '-framerate',str(framerate),
                               '-r',str(framerate),
                               '-b', str(bitrate),
                               outputfilename])
    except subprocess.CalledProcessError:
        print "'ffmpeg' failed"
    return None 

if __name__ == '__main__':
    create_frames(sys.argv[1])
    create_movie(sys.argv[1],sys.argv[2])
    

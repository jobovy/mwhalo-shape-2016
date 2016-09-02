import sys
import copy
import signal
import pickle
import numpy
import tqdm
from scipy import interpolate
from galpy.actionAngle import actionAngleIsochroneApprox, estimateBIsochrone
from galpy.actionAngle import actionAngleTorus
from galpy.orbit import Orbit
from galpy.df import streamdf
from galpy.util import bovy_conversion, bovy_coords
import MWPotential2014Likelihood
from MWPotential2014Likelihood import _REFR0, _REFV0
# Coordinate transformation routines
_RAPAL5= 229.018/180.*numpy.pi
_DECPAL5= -0.124/180.*numpy.pi
_TPAL5= numpy.dot(numpy.array([[numpy.cos(_DECPAL5),0.,numpy.sin(_DECPAL5)],
                               [0.,1.,0.],
                               [-numpy.sin(_DECPAL5),0.,numpy.cos(_DECPAL5)]]),
                  numpy.array([[numpy.cos(_RAPAL5),numpy.sin(_RAPAL5),0.],
                               [-numpy.sin(_RAPAL5),numpy.cos(_RAPAL5),0.],
                               [0.,0.,1.]]))
@bovy_coords.scalarDecorator
@bovy_coords.degreeDecorator([0,1],[0,1])
def radec_to_pal5xieta(ra,dec,degree=False):
    XYZ= numpy.array([numpy.cos(dec)*numpy.cos(ra),
                      numpy.cos(dec)*numpy.sin(ra),
                      numpy.sin(dec)])
    phiXYZ= numpy.dot(_TPAL5,XYZ)
    phi2= numpy.arcsin(phiXYZ[2])
    phi1= numpy.arctan2(phiXYZ[1],phiXYZ[0])
    return numpy.array([phi1,phi2]).T

def width_trailing(sdf):
    """Return the FWHM width in arcmin for the trailing tail"""
    # Go out to RA=245 deg
    trackRADec_trailing=\
        bovy_coords.lb_to_radec(sdf._interpolatedObsTrackLB[:,0],
                                sdf._interpolatedObsTrackLB[:,1],
                                degree=True)
    cindx= range(len(trackRADec_trailing))[\
        numpy.argmin(numpy.fabs(trackRADec_trailing[:,0]-245.))]
    ws= numpy.zeros(cindx)
    for ii,cc in enumerate(range(1,cindx+1)):
        xy= [sdf._interpolatedObsTrackLB[cc,0],None,None,None,None,None]
        ws[ii]= numpy.sqrt(sdf.gaussApprox(xy=xy,lb=True,cindx=cc)[1][0,0])
#    return 2.355*60.*ws
    return 2.355*60.*numpy.mean(ws)

def vdisp_trailing(sdf):
    """Return the velocity dispersion in km/s for the trailing tail"""
    # Go out to RA=245 deg
    trackRADec_trailing=\
        bovy_coords.lb_to_radec(sdf._interpolatedObsTrackLB[:,0],
                                sdf._interpolatedObsTrackLB[:,1],
                                degree=True)
    cindx= range(len(trackRADec_trailing))[\
        numpy.argmin(numpy.fabs(trackRADec_trailing[:,0]-245.))]
    ws= numpy.zeros(cindx)
    for ii,cc in enumerate(range(1,cindx+1)):
        xy= [sdf._interpolatedObsTrackLB[cc,0],None,None,None,None,None]
        ws[ii]= numpy.sqrt(sdf.gaussApprox(xy=xy,lb=True,cindx=cc)[1][2,2])
    return numpy.mean(ws)

def timeout_handler(signum, frame):
    raise Exception("Calculation timed-out")

def predict_pal5obs(pot_params,c,b=1.,pa=0.,
                    sigv=0.2,td=10.,
                    dist=23.2,pmra=-2.296,pmdec=-2.257,vlos=-58.7,
                    ro=_REFR0,vo=_REFV0,
                    singlec=False,
                    interpcs=None,interpk=None,
                    isob=None,nTrackChunks=8,multi=None,
                    trailing_only=False,useTM=False,
                    verbose=True):
    """
    NAME:
       predict_pal5obs
    PURPOSE:
       Function that generates the location and velocity of the Pal 5 stream, its width, and its length for a given potential and progenitor phase-space position
    INPUT:
       pot_params- array with the parameters of a potential model (see MWPotential2014Likelihood.setup_potential; only the basic parameters of the disk and halo are used, flattening is specified separately)
       c- halo flattening
       b= (1.) halo-squashed
       pa= (0.) halo PA
       sigv= (0.4) velocity dispersion in km/s (can be array of same len as interpcs)
       td= (5.) stream age in Gyr (can be array of same len as interpcs)
       dist= (23.2) progenitor distance in kpc
       pmra= (-2.296) progenitor proper motion in RA * cos(Dec) in mas/yr
       pmdec= (-2.257) progenitor proper motion in Dec in mas/yr
       vlos= (-58.7) progenitor line-of-sight velocity in km/s
       ro= (project default) distance to the GC in kpc
       vo= (project default) circular velocity at R0 in km/s
       singlec= (False) if True, just compute the observables for a single c
       interpcs= (None) values of c at which to compute the model for interpolation
       nTrackChunks= (8) nTrackChunks input to streamdf
       multi= (None) multi input to streamdf
       isob= (None) if given, b parameter of actionAngleIsochroneApprox
       trailing_only= (False) if True, only predict the trailing arm
       useTM= (True) use the TorusMapper to compute the track
       verbose= (True) print messages
    OUTPUT:
       (trailing track in RA,Dec,
        leading track in RA,Dec,
        trailing track in RA,Vlos,
        leading track in RA,Vlos,
        trailing width in arcmin,
        trailing length in deg)
        all arrays with the shape of c
    HISTORY:
       2016-06-24 - Written - Bovy (UofT)
    """
    # First compute the model for all cs at which we will interpolate
    if singlec:
        interpcs= [c]
    elif interpcs is None:
        interpcs= [0.5,0.75,1.,1.25,1.55,1.75,2.,2.25,2.5,2.75,3.]
    else:
        interpcs= copy.deepcopy(interpcs) # bc we might want to remove some
    if isinstance(sigv,float):
        sigv= [sigv for i in interpcs]
    if isinstance(td,float):
        td= [td for i in interpcs]
    if isinstance(isob,float) or isob is None:
        isob= [isob for i in interpcs]
    prog= Orbit([229.018,-0.124,dist,pmra,pmdec,vlos],
                radec=True,ro=ro,vo=vo,
                solarmotion=[-11.1,24.,7.25])
    # Setup the model
    sdf_trailing_varyc= []
    sdf_leading_varyc= []
    ii= 0
    ninterpcs= len(interpcs)
    this_useTM= copy.deepcopy(useTM)
    this_nTrackChunks= nTrackChunks
    ntries= 0
    while ii < ninterpcs:
        ic= interpcs[ii]
        pot= MWPotential2014Likelihood.setup_potential(pot_params,ic,
                                                       False,False,ro,vo,
                                                       b=b,pa=pa)
        success= True
        wentIn= ntries != 0
        # Make sure this doesn't run forever
        signal.signal(signal.SIGALRM,timeout_handler)
        signal.alarm(300)
        try:
            tsdf_trailing, tsdf_leading= setup_sdf(pot,prog,sigv[ii],td[ii],
                                                   ro,vo,multi=multi,
                                                   isob=isob[ii],
                                                   nTrackChunks=this_nTrackChunks,
                                                   trailing_only=trailing_only,
                                                   verbose=verbose,
                                                   useTM=this_useTM)
        except:
            # Catches errors and time-outs
            success= False
        signal.alarm(0)
        # Check for calc. issues
        if not success or \
                (not this_useTM and looks_funny(tsdf_trailing,tsdf_leading)):
            # Try again with TM
            this_useTM= True
            this_nTrackChunks= 21 # might as well

            #wentIn= True
            #print("Here",ntries,success)
            #sys.stdout.flush()
            
            ntries+= 1
        else:
            success= not this_useTM
            #if wentIn:
            #    print(success)
            #    sys.stdout.flush()
            ii+= 1
            # reset
            this_useTM= useTM
            this_nTrackChunks= nTrackChunks
            ntries= 0
            # Add to the list
            sdf_trailing_varyc.append(tsdf_trailing)
            sdf_leading_varyc.append(tsdf_leading)
    if not singlec and len(sdf_trailing_varyc) <= 1:
        # Almost everything bad!!
        return (numpy.zeros((len(c),1001,2)),
                numpy.zeros((len(c),1001,2)),
                numpy.zeros((len(c),1001,2)),
                numpy.zeros((len(c),1001,2)),
                numpy.zeros((len(c))),
                numpy.zeros((len(c))),[],success)
    # Compute the track properties for each model
    trackRADec_trailing= numpy.zeros((len(interpcs),
                                      sdf_trailing_varyc[0]\
                                          .nInterpolatedTrackChunks,2))
    trackRADec_leading= numpy.zeros((len(interpcs),
                                     sdf_trailing_varyc[0]\
                                         .nInterpolatedTrackChunks,2))
    trackRAVlos_trailing= numpy.zeros((len(interpcs),
                                       sdf_trailing_varyc[0]\
                                          .nInterpolatedTrackChunks,2))
    trackRAVlos_leading= numpy.zeros((len(interpcs),
                                      sdf_trailing_varyc[0]\
                                          .nInterpolatedTrackChunks,2))
    width= numpy.zeros(len(interpcs))
    length= numpy.zeros(len(interpcs))
    for ii in range(len(interpcs)):
        trackRADec_trailing[ii]= bovy_coords.lb_to_radec(\
            sdf_trailing_varyc[ii]._interpolatedObsTrackLB[:,0],
            sdf_trailing_varyc[ii]._interpolatedObsTrackLB[:,1],
            degree=True)
        if not trailing_only:
            trackRADec_leading[ii]= bovy_coords.lb_to_radec(\
                sdf_leading_varyc[ii]._interpolatedObsTrackLB[:,0],
                sdf_leading_varyc[ii]._interpolatedObsTrackLB[:,1],
                degree=True)
        trackRAVlos_trailing[ii][:,0]= trackRADec_trailing[ii][:,0]
        trackRAVlos_trailing[ii][:,1]= \
            sdf_trailing_varyc[ii]._interpolatedObsTrackLB[:,3]
        if not trailing_only:
            trackRAVlos_leading[ii][:,0]= trackRADec_leading[ii][:,0]
            trackRAVlos_leading[ii][:,1]= \
                sdf_leading_varyc[ii]._interpolatedObsTrackLB[:,3]
        width[ii]= width_trailing(sdf_trailing_varyc[ii])
        length[ii]=\
            sdf_trailing_varyc[ii].length(ang=True,coord='customra',
                                          threshold=0.3)
    if singlec:
        #if wentIn:
        #    print(success)
        #    sys.stdout.flush()
        return (trackRADec_trailing,trackRADec_leading,
                trackRAVlos_trailing,trackRAVlos_leading,
                width,length,interpcs,success)
    # Interpolate; output grids
    trackRADec_trailing_out= numpy.zeros((len(c),sdf_trailing_varyc[0]\
                                              .nInterpolatedTrackChunks,2))
    trackRADec_leading_out= numpy.zeros((len(c),sdf_trailing_varyc[0]\
                                             .nInterpolatedTrackChunks,2))
    trackRAVlos_trailing_out= numpy.zeros((len(c),sdf_trailing_varyc[0]\
                                               .nInterpolatedTrackChunks,2))
    trackRAVlos_leading_out= numpy.zeros((len(c),sdf_trailing_varyc[0]\
                                              .nInterpolatedTrackChunks,2))
    if interpk is None:
        interpk= numpy.amin([len(interpcs)-1,3])
    for ii in range(sdf_trailing_varyc[0].nInterpolatedTrackChunks):
        ip= interpolate.InterpolatedUnivariateSpline(\
            interpcs,trackRADec_trailing[:,ii,0],k=interpk,ext=0)
        trackRADec_trailing_out[:,ii,0]= ip(c)
        ip= interpolate.InterpolatedUnivariateSpline(\
            interpcs,trackRADec_trailing[:,ii,1],k=interpk,ext=0)
        trackRADec_trailing_out[:,ii,1]= ip(c)
        ip= interpolate.InterpolatedUnivariateSpline(\
            interpcs,trackRAVlos_trailing[:,ii,0],k=interpk,ext=0)
        trackRAVlos_trailing_out[:,ii,0]= ip(c)
        ip= interpolate.InterpolatedUnivariateSpline(\
            interpcs,trackRAVlos_trailing[:,ii,1],k=interpk,ext=0)
        trackRAVlos_trailing_out[:,ii,1]= ip(c)
        if not trailing_only:
            ip= interpolate.InterpolatedUnivariateSpline(\
                interpcs,trackRADec_leading[:,ii,0],k=interpk,ext=0)
            trackRADec_leading_out[:,ii,0]= ip(c)
            ip= interpolate.InterpolatedUnivariateSpline(\
                interpcs,trackRADec_leading[:,ii,1],k=interpk,ext=0)
            trackRADec_leading_out[:,ii,1]= ip(c)
            ip= interpolate.InterpolatedUnivariateSpline(\
                interpcs,trackRAVlos_leading[:,ii,0],k=interpk,ext=0)
            trackRAVlos_leading_out[:,ii,0]= ip(c)
            ip= interpolate.InterpolatedUnivariateSpline(\
                interpcs,trackRAVlos_leading[:,ii,1],k=interpk,ext=0)
            trackRAVlos_leading_out[:,ii,1]= ip(c)
    ip= interpolate.InterpolatedUnivariateSpline(\
        interpcs,width,k=interpk,ext=0)
    width_out= ip(c)
    ip= interpolate.InterpolatedUnivariateSpline(\
        interpcs,length,k=interpk,ext=0)
    length_out= ip(c)
    return (trackRADec_trailing_out,trackRADec_leading_out,
            trackRAVlos_trailing_out,trackRAVlos_leading_out,
            width_out,length_out,interpcs,success)

def looks_funny(tsdf_trailing,tsdf_leading):
    radecs_trailing=\
        bovy_coords.lb_to_radec(tsdf_trailing._interpolatedObsTrackLB[:,0],
                                tsdf_trailing._interpolatedObsTrackLB[:,1],
                                degree=True)
    if not tsdf_leading is None:
        radecs_leading=\
            bovy_coords.lb_to_radec(tsdf_leading._interpolatedObsTrackLB[:,0],
                                    tsdf_leading._interpolatedObsTrackLB[:,1],
                                    degree=True)
    try:
        if radecs_trailing[0,1] > 0.625:
            return True
        elif radecs_trailing[0,1] < -0.1:
            return True
        elif numpy.any((numpy.roll(radecs_trailing[:,0],-1)-radecs_trailing[:,0])\
                           [(radecs_trailing[:,0] < 250.)\
                                *(radecs_trailing[:,1] > -1.)\
                                *(radecs_trailing[:,1] < 10.)] < 0.):
            return True
        elif not tsdf_leading is None and \
                numpy.any((numpy.roll(radecs_leading[:,0],-1)-radecs_leading[:,0])\
                              [(radecs_leading[:,0] > 225.)\
                                   *(radecs_leading[:,1] > -4.5)\
                                   *(radecs_leading[:,1] < 0.)] > 0.):
            return True
        elif False:#numpy.isnan(width_trailing(tsdf_trailing)):
            return True
        elif numpy.isnan(tsdf_trailing.length(ang=True,coord='customra',
                                              threshold=0.3)):
            return True
        elif numpy.fabs(tsdf_trailing._dOdJpEig[0][2]\
                            /tsdf_trailing._dOdJpEig[0][1]) < 0.05:
            return True
        else:
            return False
    except:
        return True

def pal5_lnlike(pos_radec,rvel_ra,
                trackRADec_trailing,trackRADec_leading,
                trackRAVlos_trailing,trackRAVlos_leading,
                width_out,length_out,interpcs): # last one so we can do *args
    """
    Returns array [nmodel,5] with log likelihood for each a) model, b) data set (trailing position, leading position, trailing vlos, actual width, actual length)
    """
    nmodel= trackRADec_trailing.shape[0]
    out= numpy.zeros((nmodel,5))-1000000000000000.
    for nn in range(nmodel):
        # Interpolate trailing RA,Dec track
        sindx= numpy.argsort(trackRADec_trailing[nn,:,0])
        ipdec= \
            interpolate.InterpolatedUnivariateSpline(\
            trackRADec_trailing[nn,sindx,0],trackRADec_trailing[nn,sindx,1],
            k=1) # to be on the safe side
        tindx= pos_radec[:,0] > 229.
        out[nn,0]= \
            -0.5*numpy.sum((ipdec(pos_radec[tindx,0])-pos_radec[tindx,1])**2.\
                              /pos_radec[tindx,2]**2.)
        # Interpolate leading RA,Dec track
        sindx= numpy.argsort(trackRADec_leading[nn,:,0])
        ipdec= \
            interpolate.InterpolatedUnivariateSpline(\
            trackRADec_leading[nn,sindx,0],trackRADec_leading[nn,sindx,1],
            k=1) # to be on the safe side
        tindx= pos_radec[:,0] < 229.
        out[nn,1]= \
            -0.5*numpy.sum((ipdec(pos_radec[tindx,0])-pos_radec[tindx,1])**2.\
                              /pos_radec[tindx,2]**2.)
        # Interpolate trailing RA,Vlos track
        sindx= numpy.argsort(trackRAVlos_trailing[nn,:,0])
        ipvlos= \
            interpolate.InterpolatedUnivariateSpline(\
            trackRAVlos_trailing[nn,sindx,0],trackRAVlos_trailing[nn,sindx,1],
            k=1) # to be on the safe side
        tindx= rvel_ra[:,0] > 230.5
        out[nn,2]= \
            -0.5*numpy.sum((ipvlos(rvel_ra[tindx,0])-rvel_ra[tindx,1])**2.\
                              /rvel_ra[tindx,2]**2.)
        out[nn,3]= width_out[nn]
        out[nn,4]= length_out[nn]
    out[numpy.isnan(out[:,0]),0]= -1000000000000000000.
    out[numpy.isnan(out[:,1]),1]= -1000000000000000000.
    out[numpy.isnan(out[:,2]),2]= -1000000000000000000.
    return out

def setup_sdf(pot,prog,sigv,td,ro,vo,multi=None,nTrackChunks=8,isob=None,
              trailing_only=False,verbose=True,useTM=True):
    if isob is None:
        # Determine good one
        ts= numpy.linspace(0.,150.,1001)
        # Hack!
        epot= copy.deepcopy(pot)
        epot[2]._b= 1.
        epot[2]._b2= 1.
        epot[2]._isNonAxi= False
        epot[2]._aligned= True
        prog.integrate(ts,pot)
        estb= estimateBIsochrone(epot,
                                 prog.R(ts,use_physical=False),
                                 prog.z(ts,use_physical=False),
                                 phi=prog.phi(ts,use_physical=False))
        if estb[1] < 0.3: isob= 0.3
        elif estb[1] > 1.5: isob= 1.5
        else: isob= estb[1]
    if verbose: print(pot[2]._c, isob)
    if numpy.fabs(pot[2]._b-1.) > 0.05:
        aAI= actionAngleIsochroneApprox(pot=pot,b=isob,tintJ=1000.,
                                        ntintJ=30000)
    else:
        aAI= actionAngleIsochroneApprox(pot=pot,b=isob)
    if useTM:
        aAT= actionAngleTorus(pot=pot,tol=0.001,dJ=0.0001)
    else:
        aAT= False
    try:
        sdf_trailing=\
            streamdf(sigv/vo,progenitor=prog,pot=pot,aA=aAI,
                     useTM=aAT,approxConstTrackFreq=True,
                     leading=False,nTrackChunks=nTrackChunks,
                     tdisrupt=td/bovy_conversion.time_in_Gyr(vo,ro),
                     ro=ro,vo=vo,R0=ro,
                     vsun=[-11.1,vo+24.,7.25],
                     custom_transform=_TPAL5,
                     multi=multi)
    except numpy.linalg.LinAlgError:
        sdf_trailing=\
            streamdf(sigv/vo,progenitor=prog,pot=pot,aA=aAI,
                     useTM=aAT,approxConstTrackFreq=True,
                     leading=False,nTrackChunks=nTrackChunks,
                     nTrackIterations=0,
                     tdisrupt=td/bovy_conversion.time_in_Gyr(vo,ro),
                     ro=ro,vo=vo,R0=ro,
                     vsun=[-11.1,vo+24.,7.25],
                     custom_transform=_TPAL5,
                     multi=multi)
    if trailing_only:
        return (sdf_trailing,None)
    try:
        sdf_leading=\
            streamdf(sigv/vo,progenitor=prog,pot=pot,aA=aAI,
                     useTM=aAT,approxConstTrackFreq=True,
                     leading=True,nTrackChunks=nTrackChunks,
                     tdisrupt=td/bovy_conversion.time_in_Gyr(vo,ro),
                     ro=ro,vo=vo,R0=ro,
                     vsun=[-11.1,vo+24.,7.25],
                     custom_transform=_TPAL5,
                     multi=multi)
    except numpy.linalg.LinAlgError:
        sdf_leading=\
            streamdf(sigv/vo,progenitor=prog,pot=pot,aA=aAI,
                     useTM=aAT,approxConstTrackFreq=True,
                     leading=True,nTrackChunks=nTrackChunks,
                     nTrackIterations=0,
                     tdisrupt=td/bovy_conversion.time_in_Gyr(vo,ro),
                     ro=ro,vo=vo,R0=ro,
                     vsun=[-11.1,vo+24.,7.25],
                     custom_transform=_TPAL5,
                     multi=multi)
    return (sdf_trailing,sdf_leading)

def pal5_dpmguess(pot,doras=None,dodecs=None,dovloss=None,
                  dmin=21.,dmax=25.,dstep=0.02,
                  pmmin=-0.36,pmmax=0.36,pmstep=0.01,
                  alongbfpm=False,
                  ro=_REFR0,vo=_REFV0):
    if doras is None:
        with open('pal5_stream_orbit_offset.pkl','rb') as savefile:
            doras= pickle.load(savefile)
            dodecs= pickle.load(savefile)
            dovloss= pickle.load(savefile)
    ds= numpy.arange(dmin,dmax+dstep/2.,dstep)
    pmoffs= numpy.arange(pmmin,pmmax+pmstep/2.,pmstep)
    lnl= numpy.zeros((len(ds),len(pmoffs)))
    pos_radec, rvel_ra= pal5_data()
    print("Determining good distance and parallel proper motion...")
    for ii,d in tqdm.tqdm(enumerate(ds)):
        for jj,pmoff in enumerate(pmoffs):
            if alongbfpm:
                pm= (d-24.)*0.099+0.0769+pmoff
            else:
                pm= pmoff
            progt= Orbit([229.11,0.3,d+0.3,
                          -2.27+pm,-2.22+pm*2.257/2.296,-58.5],
                         radec=True,ro=ro,vo=vo,
                         solarmotion=[-11.1,24.,7.25]).flip()
            ts= numpy.linspace(0.,3.,1001)
            progt.integrate(ts,pot)
            progt._orb.orbit[:,1]*= -1.
            progt._orb.orbit[:,2]*= -1.
            progt._orb.orbit[:,4]*= -1.
            toras, todecs, tovloss= progt.ra(ts), progt.dec(ts), progt.vlos(ts)
            # Interpolate onto common RA
            ipdec= interpolate.InterpolatedUnivariateSpline(toras,todecs)
            ipvlos= interpolate.InterpolatedUnivariateSpline(toras,tovloss)
            todecs= ipdec(doras)-dodecs
            tovloss= ipvlos(doras)-dovloss
            est_trackRADec_trailing= numpy.zeros((1,len(doras),2))
            est_trackRADec_trailing[0,:,0]= doras
            est_trackRADec_trailing[0,:,1]= todecs
            est_trackRAVlos_trailing= numpy.zeros((1,len(doras),2))
            est_trackRAVlos_trailing[0,:,0]= doras
            est_trackRAVlos_trailing[0,:,1]= tovloss
            lnl[ii,jj]= numpy.sum(\
                pal5_lnlike(pos_radec,rvel_ra,
                            est_trackRADec_trailing,
                            est_trackRADec_trailing, # hack
                            est_trackRAVlos_trailing,
                            est_trackRAVlos_trailing, # hack
                            numpy.array([0.]),
                            numpy.array([0.]),None)[0,:3:2])\
                                    -0.5*pm**2./0.186**2. #pm measurement
    bestd= ds[numpy.unravel_index(numpy.argmax(lnl),lnl.shape)[0]]
    if alongbfpm:
        bestpmoff= (bestd-24.)*0.099+0.0769\
            +pmoffs[numpy.unravel_index(numpy.argmax(lnl),lnl.shape)[1]]
    else:
        bestpmoff= pmoffs[numpy.unravel_index(numpy.argmax(lnl),lnl.shape)[1]]
    return (bestd,bestpmoff,lnl,ds,pmoffs)

def pal5_data():
    pos_radec= numpy.array([[241.48,6.41,0.09],
                            [240.98,6.15,0.09],
                            [240.48,6.20,0.09],
                            [239.98,5.81,0.09],
                            [239.48,5.64,0.09],
                            [238.48,5.38,0.09],
                            [237.98,5.14,0.09],
                            [233.61,3.17,0.06],
                            [233.11,2.88,0.06],
                            [232.61,2.54,0.06],
                            [232.11,2.23,0.06],
                            [231.61,2.04,0.06],
                            [231.11,1.56,0.06],
                            [230.11,0.85,0.06],
                            [229.61,0.54,0.06],
                            [228.48,-0.77,0.11],
                            [228.11,-1.16,0.14],
                            [227.73,-1.28,0.11],
                            [227.23,-2.03,0.17],
                            [226.55,-2.59,0.14]])
    rvel_ra= numpy.array([[225+15*15/60+48.19*0.25/60,-55.9,1.2],
                          [225+15*15/60+49.70*0.25/60,-56.9,0.4],
                          [225+15*15/60+52.60*0.25/60,-56.0,0.6],
                          [225+15*15/60+54.79*0.25/60,-57.6,1.6],
                          [225+15*15/60+56.11*0.25/60,-57.9,0.7],
                          [225+15*15/60+57.05*0.25/60,-55.6,1.5],
                          [225+15*15/60+58.26*0.25/60,-56.4,1.0],
                          [225+15*15/60+58.89*0.25/60,-55.9,0.3],
                          [225+15*15/60+59.52*0.25/60,-59.0,0.4],
                          [225+16*15/60+02.00*0.25/60,-58.0,0.8],
                          [225+16*15/60+03.61*0.25/60,-57.7,2.5],
                          [225+16*15/60+04.81*0.25/60,-57.2,2.7],
                          [225+16*15/60+06.54*0.25/60,-57.1,0.2],
                          [225+16*15/60+07.75*0.25/60,-60.6,0.3],
                          [225+16*15/60+08.51*0.25/60,-60.9,3.3],
                          [225+16*15/60+19.83*0.25/60,-56.9,1.0],
                          [225+16*15/60+23.11*0.25/60,-58.0,2.5],
                          [225+16*15/60+34.71*0.25/60,-58.2,3.8],
                          [225+16*15/60+08.66*0.25/60,-56.8,0.7],
                          [225+16*15/60+09.58*0.25/60,-57.7,0.3],
                          [225+15*15/60+52.84*0.25/60,-55.7,0.6],
                          [225+15*15/60+56.21*0.25/60,-55.9,0.7],
                          [225+16*15/60+05.26*0.25/60,-54.3,0.3],
                          [225+17*15/60+09.99*0.25/60,-57.0,0.4],
                          [225+17*15/60+34.55*0.25/60,-56.5,3.1],
                          [225+17*15/60+58.32*0.25/60,-57.5,3.3],
                          [225+18*15/60+04.96*0.25/60,-57.7,2.6],
                          [225+18*15/60+18.92*0.25/60,-57.6,3.6],
                          [225+18*15/60+35.89*0.25/60,-56.7,1.3],
                          [225+19*15/60+21.42*0.25/60,-61.7,3.1],
                          [225+21*15/60+51.16*0.25/60,-55.6,0.4],
                          [225+24*15/60+04.85*0.25/60,-56.5,2.6],
                          [225+24*15/60+13.00*0.25/60,-50.0,2.4],
                          [225+28*15/60+39.20*0.25/60,-56.6,1.4],
                          [225+28*15/60+49.34*0.25/60,-52.4,3.8],
                          [225+34*15/60+19.31*0.25/60,-55.8,1.8],
                          [225+34*15/60+31.90*0.25/60,-52.7,4.0],
                          [225+34*15/60+56.51*0.25/60,-51.9,1.6],
                          [225+45*15/60+10.57*0.25/60,-45.6,2.6],
                          [225+46*15/60+49.44*0.25/60,-48.0,2.4],
                          [225+48*15/60+57.99*0.25/60,-46.7,2.3],
                          [225+55*15/60+24.13*0.25/60,-41.0,2.7],
                          [240+0*15/60+45.41*0.25/60,-41.1,2.8],
                          [240+1*15/60+12.59*0.25/60,-40.8,2.5],
                          [240+3*15/60+29.59*0.25/60,-45.2,3.9],
                          [240+4*15/60+05.53*0.25/60,-44.9,4.0],
                          [240+4*15/60+33.28*0.25/60,-45.1,3.5],
                          [240+13*15/60+40.97*0.25/60,-41.1,3.4],
                          [240+16*15/60+44.79*0.25/60,-44.0,3.0],
                          [240+16*15/60+51.73*0.25/60,-43.5,2.5],
                          [225+8*15/60+07.15*0.25/60,-57.8,1.1],
                          [225+8*15/60+17.50*0.25/60,-62.0,2.3],
                          [225+10*15/60+39.02*0.25/60,-58.0,1.0],
                          [225+11*15/60+09.04*0.25/60,-66.9,2.1],
                          [225+11*15/60+21.70*0.25/60,-53.8,1.1],
                          [225+12*15/60+45.44*0.25/60,-52.5,2.2],
                          [225+13*15/60+40.44*0.25/60,-58.6,1.4],
                          [225+13*15/60+54.40*0.25/60,-59.8,3.7],
                          [225+14*15/60+09.32*0.25/60,-57.9,3.5],
                          [225+14*15/60+17.18*0.25/60,-59.2,1.7],
                          [225+14*15/60+20.71*0.25/60,-56.7,2.3],
                          [225+14*15/60+34.63*0.25/60,-59.1,1.3],
                          [225+15*15/60+16.47*0.25/60,-58.6,2.3],
                          [225+15*15/60+50.43*0.25/60,-55.7,2.3],
                          [225+16*15/60+01.54*0.25/60,-58.7,1.4],
                          [225+16*15/60+34.95*0.25/60,-59.7,0.4],
                          [225+16*15/60+56.20*0.25/60,-58.7,0.2]])
    return (pos_radec,rvel_ra)

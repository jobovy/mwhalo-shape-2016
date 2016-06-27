import copy
import warnings
import numpy
from scipy import interpolate
from galpy.actionAngle import actionAngleIsochroneApprox
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

def predict_pal5obs(pot_params,c,
                    sigv=0.4,td=5.,
                    dist=23.2,pmra=-2.296,pmdec=-2.257,vlos=-58.7,
                    ro=_REFR0,vo=_REFV0,
                    singlec=False,
                    interpcs=None,interpk=None,
                    nTrackChunks=8,multi=None):
    """
    NAME:
       predict_pal5obs
    PURPOSE:
       Function that generates the location and velocity of the Pal 5 stream, its width, and its length for a given potential and progenitor phase-space position
    INPUT:
       pot_params- array with the parameters of a potential model (see MWPotential2014Likelihood.setup_potential; only the basic parameters of the disk and halo are used, flattening is specified separately)
       c- halo flattening
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
    # Setup the model
    sdf_trailing_varyc= []
    sdf_leading_varyc= []
    for ii,ic in enumerate(interpcs):
        pot= MWPotential2014Likelihood.setup_potential(pot_params,ic,
                                                       False,False,ro,vo)
        prog= Orbit([229.018,-0.124,dist,pmra,pmdec,vlos],
                    radec=True,ro=ro,vo=vo,
                    solarmotion=[-11.1,24.,7.25])
        aAI= actionAngleIsochroneApprox(pot=pot,b=0.8)
        try:
            tsdf_trailing=\
                streamdf(sigv[ii]/vo,progenitor=prog,pot=pot,aA=aAI,
                         leading=False,nTrackChunks=nTrackChunks,
                         tdisrupt=td[ii]/bovy_conversion.time_in_Gyr(vo,ro),
                         ro=ro,vo=vo,R0=ro,
                         vsun=[-11.1,vo+24.,7.25],
                         custom_transform=_TPAL5,
                         multi=multi)
        except numpy.linalg.LinAlgError:
            tsdf_trailing=\
                streamdf(sigv[ii]/vo,progenitor=prog,pot=pot,aA=aAI,
                         leading=False,nTrackChunks=nTrackChunks,
                         nTrackIterations=0,
                         tdisrupt=td[ii]/bovy_conversion.time_in_Gyr(vo,ro),
                         ro=ro,vo=vo,R0=ro,
                         vsun=[-11.1,vo+24.,7.25],
                         custom_transform=_TPAL5,
                         multi=multi)
        try:
            tsdf_leading=\
                streamdf(sigv[ii]/vo,progenitor=prog,pot=pot,aA=aAI,
                         leading=True,nTrackChunks=nTrackChunks,
                         tdisrupt=td[ii]/bovy_conversion.time_in_Gyr(vo,ro),
                         ro=ro,vo=vo,R0=ro,
                         vsun=[-11.1,vo+24.,7.25],
                         custom_transform=_TPAL5,
                         multi=multi)
        except numpy.linalg.LinAlgError:
            tsdf_leading=\
                streamdf(sigv[ii]/vo,progenitor=prog,pot=pot,aA=aAI,
                         leading=True,nTrackChunks=nTrackChunks,
                         nTrackIterations=0,
                         tdisrupt=td[ii]/bovy_conversion.time_in_Gyr(vo,ro),
                         ro=ro,vo=vo,R0=ro,
                         vsun=[-11.1,vo+24.,7.25],
                         custom_transform=_TPAL5,
                         multi=multi)
        # Check that we're not too close to having a problem
        if numpy.fabs(tsdf_trailing._dOdJpEig[0][2]\
                          /tsdf_trailing._dOdJpEig[0][1]) < 0.05:
            warnings.warn("Potential with c = %.1f close to one for which we expect there to be problems with the calculation" % ic)
            if not singlec:
                interpcs.remove(ic)
                continue
        # Add to the list
        sdf_trailing_varyc.append(tsdf_trailing)
        sdf_leading_varyc.append(tsdf_leading)
    # Compute the track properties for each model
    trackRADec_trailing= numpy.empty((len(interpcs),
                                      sdf_trailing_varyc[0]\
                                          .nInterpolatedTrackChunks,2))
    trackRADec_leading= numpy.empty((len(interpcs),
                                     sdf_trailing_varyc[0]\
                                         .nInterpolatedTrackChunks,2))
    trackRAVlos_trailing= numpy.empty((len(interpcs),
                                       sdf_trailing_varyc[0]\
                                          .nInterpolatedTrackChunks,2))
    trackRAVlos_leading= numpy.empty((len(interpcs),
                                      sdf_trailing_varyc[0]\
                                          .nInterpolatedTrackChunks,2))
    width= numpy.empty(len(interpcs))
    length= numpy.empty(len(interpcs))
    for ii in range(len(interpcs)):
        trackRADec_trailing[ii]= bovy_coords.lb_to_radec(\
            sdf_trailing_varyc[ii]._interpolatedObsTrackLB[:,0],
            sdf_trailing_varyc[ii]._interpolatedObsTrackLB[:,1],
            degree=True)
        trackRADec_leading[ii]= bovy_coords.lb_to_radec(\
            sdf_leading_varyc[ii]._interpolatedObsTrackLB[:,0],
            sdf_leading_varyc[ii]._interpolatedObsTrackLB[:,1],
            degree=True)
        trackRAVlos_trailing[ii][:,0]= trackRADec_trailing[ii][:,0]
        trackRAVlos_trailing[ii][:,1]= \
            sdf_trailing_varyc[ii]._interpolatedObsTrackLB[:,3]
        trackRAVlos_leading[ii][:,0]= trackRADec_leading[ii][:,0]
        trackRAVlos_leading[ii][:,1]= \
            sdf_leading_varyc[ii]._interpolatedObsTrackLB[:,3]
        width[ii]= width_trailing(sdf_trailing_varyc[ii])
        length[ii]=\
            sdf_trailing_varyc[ii].length(ang=True,coord='customra',
                                          threshold=0.3)
    if singlec:
        return (trackRADec_trailing,trackRADec_leading,
                trackRAVlos_trailing,trackRAVlos_leading,
                width,length)
    # Interpolate; output grids
    trackRADec_trailing_out= numpy.empty((len(c),sdf_trailing_varyc[0]\
                                              .nInterpolatedTrackChunks,2))
    trackRADec_leading_out= numpy.empty((len(c),sdf_trailing_varyc[0]\
                                             .nInterpolatedTrackChunks,2))
    trackRAVlos_trailing_out= numpy.empty((len(c),sdf_trailing_varyc[0]\
                                               .nInterpolatedTrackChunks,2))
    trackRAVlos_leading_out= numpy.empty((len(c),sdf_trailing_varyc[0]\
                                              .nInterpolatedTrackChunks,2))
    if interpk is None:
        interpk= numpy.amin([len(interpcs)-1,3])
    for ii in range(sdf_trailing_varyc[0].nInterpolatedTrackChunks):
        ip= interpolate.InterpolatedUnivariateSpline(\
            interpcs,trackRADec_trailing[:,ii,0],k=interpk)
        trackRADec_trailing_out[:,ii,0]= ip(c)
        ip= interpolate.InterpolatedUnivariateSpline(\
            interpcs,trackRADec_trailing[:,ii,1],k=interpk)
        trackRADec_trailing_out[:,ii,1]= ip(c)
        ip= interpolate.InterpolatedUnivariateSpline(\
            interpcs,trackRAVlos_trailing[:,ii,0],k=interpk)
        trackRAVlos_trailing_out[:,ii,0]= ip(c)
        ip= interpolate.InterpolatedUnivariateSpline(\
            interpcs,trackRAVlos_trailing[:,ii,1],k=interpk)
        trackRAVlos_trailing_out[:,ii,1]= ip(c)
        ip= interpolate.InterpolatedUnivariateSpline(\
            interpcs,trackRADec_leading[:,ii,0],k=interpk)
        trackRADec_leading_out[:,ii,0]= ip(c)
        ip= interpolate.InterpolatedUnivariateSpline(\
            interpcs,trackRADec_leading[:,ii,1],k=interpk)
        trackRADec_leading_out[:,ii,1]= ip(c)
        ip= interpolate.InterpolatedUnivariateSpline(\
            interpcs,trackRAVlos_leading[:,ii,0],k=interpk)
        trackRAVlos_leading_out[:,ii,0]= ip(c)
        ip= interpolate.InterpolatedUnivariateSpline(\
            interpcs,trackRAVlos_leading[:,ii,1],k=interpk)
        trackRAVlos_leading_out[:,ii,1]= ip(c)
    ip= interpolate.InterpolatedUnivariateSpline(\
        interpcs,width,k=interpk)
    width_out= ip(c)
    ip= interpolate.InterpolatedUnivariateSpline(\
        interpcs,length,k=interpk)
    length_out= ip(c)
    return (trackRADec_trailing_out,trackRADec_leading_out,
            trackRAVlos_trailing_out,trackRAVlos_leading_out,
            width_out,length_out)

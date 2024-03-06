from __future__ import print_function, division

import numpy as np

from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_transpose
import astropy.coordinates as coord, astropy.units as u
from astropy.io import fits
from astropy import wcs

def getMagellanicWCS(xe,ye):
    head = fits.PrimaryHDU().header
    head['NAXIS1'] = len(xe)
    head['NAXIS2'] = len(ye)
    head['CRPIX1'] = len(xe)/2
    head['CRPIX2'] = len(ye)/2
    head['CDELT1'] = xe[1]-xe[0]
    head['CDELT2'] = ye[1]-ye[0]
    head['CUNIT1'] = 'deg'
    head['CUNIT2'] = 'deg'
    head['CRVAL1'] = (xe[0]+xe[-1])/2.0
    head['CRVAL2'] = (ye[0]+ye[-1])/2.0

    head['WCSAXES'] = 2
    head['NAXIS'] = 2

    return wcs.WCS(head)

def getGalacticWCS(lmcloc,xe,ye):
    head = fits.PrimaryHDU().header
    head['CRPIX1'] = lmcloc[0]
    head['CRPIX2'] = lmcloc[1]
    head['CDELT1'] = xe[1]-xe[0]
    head['CDELT2'] = ye[1]-ye[0]
    head['CUNIT1'] = 'deg'
    head['CUNIT2'] = 'deg'
    head['CTYPE1'] = 'GLON-CYP'
    head['CTYPE2'] = 'GLAT-CYP'
    lmcgalc = coord.SkyCoord.from_name("LMC").transform_to(coord.Galactic)
    head['CRVAL1'] = lmcgalc.l.value
    head['CRVAL2'] = lmcgalc.b.value
    head['LONPOLE'] = 123.3
    head['LATPOLE'] = -800

    head['WCSAXES'] = 2
    head['NAXIS'] = 2

    return wcs.WCS(head)

def getEquatorialWCS(lmcloc,xe,ye):
    head = fits.PrimaryHDU().header
    head['CRPIX1'] = lmcloc[0]
    head['CRPIX2'] = lmcloc[1]
    head['CDELT1'] = xe[1]-xe[0]
    head['CDELT2'] = ye[1]-ye[0]
    head['CUNIT1'] = 'deg'
    head['CUNIT2'] = 'deg'
    head['CTYPE1'] = 'RA---CYP'
    head['CTYPE2'] = 'DEC--CYP'
    lmcgalc = coord.SkyCoord.from_name("LMC").transform_to(coord.ICRS)
    head['CRVAL1'] = lmcgalc.ra.value
    head['CRVAL2'] = lmcgalc.dec.value
    # head['LONPOLE'] = 123.3
    # head['LATPOLE'] = -800

    head['WCSAXES'] = 2
    head['NAXIS'] = 2

    return wcs.WCS(head)

class MagellanicStream(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the Magellanic Stream

    Parameters
    ----------
    representation : `BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)

    MSLongitude : `Angle`, optional, must be keyword
        The longitude-like angle corresponding to the Magellanic Stream.
    MSLatitude : `Angle`, optional, must be keyword
        The latitude-like angle corresponding to the Magellanic Stream.
    distance : `Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.

    pm_Lambda_cosBeta : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion along the Stream in ``Lambda`` (including the
        ``cos(Beta)`` factor) for this object (``pm_Beta`` must also be given).
    pm_Beta : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in Declination for this object (``pm_ra_cosdec`` must
        also be given).
    radial_velocity : :class:`~astropy.units.Quantity`, optional, must be keyword
        The radial velocity of this object.

    """
    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'MSLongitude'),
            coord.RepresentationMapping('lat', 'MSLatitude'),
            coord.RepresentationMapping('distance', 'distance')],
        coord.SphericalCosLatDifferential: [
            coord.RepresentationMapping('d_lon_coslat', 'pm_lon_coslat'),
            coord.RepresentationMapping('d_lat', 'pm_lat'),
            coord.RepresentationMapping('d_distance', 'radial_velocity')],
        coord.SphericalDifferential: [
            coord.RepresentationMapping('d_lon', 'pm_lon'),
            coord.RepresentationMapping('d_lat', 'pm_lat'),
            coord.RepresentationMapping('d_distance', 'radial_velocity')]
    }

    frame_specific_representation_info[coord.UnitSphericalRepresentation] = \
        frame_specific_representation_info[coord.SphericalRepresentation]
    frame_specific_representation_info[coord.UnitSphericalCosLatDifferential] = \
        frame_specific_representation_info[coord.SphericalCosLatDifferential]
    frame_specific_representation_info[coord.UnitSphericalDifferential] = \
        frame_specific_representation_info[coord.SphericalDifferential]


MS_PHI = (180 + 8.5 + 90) * u.degree # Euler angles (from Nidever 2010)
MS_THETA = (90 + 7.5) * u.degree
MS_PSI = -32.724214217871349 * u.degree  # anode parameter from gal2mag.pro

D = rotation_matrix(MS_PHI, "z")
C = rotation_matrix(MS_THETA, "x")
B = rotation_matrix(MS_PSI, "z")
A = np.diag([1., 1., 1.])
MS_MATRIX = A @ B @ C @ D

@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, MagellanicStream)
def galactic_to_MS():
    """ Compute the transformation matrix from Galactic spherical to
        Magellanic Stream coordinates.
    """
    return MS_MATRIX

@frame_transform_graph.transform(coord.StaticMatrixTransform, MagellanicStream, coord.Galactic)
def MS_to_galactic():
    """ Compute the transformation matrix from Magellanic Stream coordinates to
        spherical Galactic.
    """
    return matrix_transpose(MS_MATRIX)

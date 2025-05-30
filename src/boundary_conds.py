from shapely.geometry import Point
import cartopy.feature as cfeature
import deepxde as dde


def is_in_ocean(lat, lon, coastline):
    point = Point(lon, lat)
    for geometry in coastline.geometries():
        if geometry.contains(point):
            return False
    return True


# Currently not working because `coastline` i a global variable
# and can't be an argument for the `boundary_condition` function
# and python is weird
# def boundary_condition(x, on_boundary):
#     lat = x[0]
#     lon = x[1]
#     ocean_boundary = is_in_ocean(lat, lon, coastline)
#     return on_boundary and ocean_boundary


def get_xt_geom(lat, lon, time):
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    time_min, time_max = time.min(), time.max()
    spatial_domain = dde.geometry.Rectangle(
        xmin=[lat_min, lon_min], xmax=[lat_max, lon_max]
    )
    temporal_domain = dde.geometry.TimeDomain(t0=time_min, t1=time_max)
    geomtime = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)
    coastline = cfeature.NaturalEarthFeature("physical", "coastline", "50m")

    return geomtime, coastline

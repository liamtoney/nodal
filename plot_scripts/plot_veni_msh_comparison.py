import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
from matplotlib.colors import LightSource
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from obspy.geodetics.base import kilometer2degrees
from pyproj import CRS, Transformer
from rasterio.enums import Resampling

URL_MSH = 'http://service.iris.edu/fdsnws/station/1/query?net=1D&cha=DPZ&maxlat=46.8&maxlon=-121.3&minlat=45.6&minlon=-123.1&format=geocsv'
URL_VENI = 'http://service.iris.edu/fdsnws/station/1/query?net=AV&cha=BDF&sta=S14K,VN??&format=geocsv'

RADIUS = 15  # [km] From center of stations


def get_dem_and_stations(iris_url):

    station_df = pd.read_csv(iris_url, sep='|', comment='#')

    mean_lon = station_df.Longitude.mean()
    mean_lat = station_df.Latitude.mean()
    radius_deg = kilometer2degrees(RADIUS * 2)  # Adding buffer here!

    # Get DEM
    region = [
        mean_lon - radius_deg,
        mean_lon + radius_deg,
        mean_lat - radius_deg,
        mean_lat + radius_deg,
    ]
    with pygmt.config(GMT_VERBOSE='e'):  # Suppress warnings
        dem = pygmt.datasets.load_earth_relief(
            resolution='01s', region=region, use_srtm=True
        )
    dem.rio.write_crs(dem.horizontal_datum, inplace=True)

    # Project DEM to UTM
    utm_crs = CRS(dem.rio.estimate_utm_crs(datum_name='WGS 84'))
    dem_utm = dem.rio.reproject(utm_crs, resampling=Resampling.cubic_spline).drop(
        'spatial_ref'
    )

    # Get UTM coords for receivers
    proj = Transformer.from_crs(utm_crs.geodetic_crs, utm_crs)
    rec_xs, rec_ys = proj.transform(station_df.Latitude, station_df.Longitude)

    return dem_utm, rec_xs, rec_ys, station_df.Elevation.to_numpy()


def plot_dem_and_stations(dem_utm, rec_xs, rec_ys, rec_elevs):

    fig, ax = plt.subplots()

    dem_utm['x'] = dem_utm.x - rec_xs.mean()
    dem_utm['y'] = dem_utm.y - rec_ys.mean()

    # Create hillshade
    ls = LightSource()
    hs = dem_utm.copy()
    hs.data = ls.hillshade(
        dem_utm.data,
        dx=np.abs(np.diff(dem_utm.x).mean()),
        dy=np.abs(np.diff(dem_utm.y).mean()),
    )

    # Plot
    hs.plot.imshow(ax=ax, cmap='Greys_r', vmin=0.2, add_colorbar=False, alpha=0.4)
    sort_ind = rec_elevs.argsort()  # Place lower-elev markers below higher-elev ones
    ax.scatter(
        rec_xs[sort_ind] - rec_xs.mean(),
        rec_ys[sort_ind] - rec_ys.mean(),
        color='tab:red',
        edgecolor='black',
        linewidth=0.5,
    )
    ax.set_aspect('equal')
    ax.axis('off')
    radius_m = RADIUS * 1000
    ax.set_xlim(-radius_m, radius_m)
    ax.set_ylim(-radius_m, radius_m)

    # Common annotation values
    linewidth = 20  # [m]
    y = 0.03
    kwargs = dict(
        pad=0,
        borderpad=0,
        sep=5,
        frameon=False,
        transform=ax.transData,
        bbox_transform=ax.transAxes,
    )

    # Add scalebar
    bar_length = 5000  # [m]
    scalebar = AnchoredSizeBar(
        size=bar_length,  # [m]
        label=f'{bar_length / 1000:g} km',
        loc='lower left',
        size_vertical=linewidth,  # [m]
        bbox_to_anchor=(0.5, y),
        **kwargs,
    )
    ax.add_artist(scalebar)

    # North arrow (HACKY!)
    arrow_x = 0.94
    arrow = AnchoredSizeBar(
        size=linewidth,  # [m]
        label='N',
        loc='lower center',
        size_vertical=1000,  # [m] Hard-coded
        bbox_to_anchor=(arrow_x, y),
        **kwargs,
    )
    ax.add_artist(arrow)
    arrow_head = FancyArrowPatch(
        (arrow_x, 0.1),  # Hard-coded
        (arrow_x, 0.14),  # Hard-coded
        arrowstyle='wedge,tail_width=6',
        linewidth=0,
        facecolor='black',
        transform=ax.transAxes,
    )
    ax.add_artist(arrow_head)

    width_km = RADIUS * 2
    print(f'{width_km:g} km x {width_km:g} km')

    fig.show()

    return fig


veni_dem_stations = get_dem_and_stations(URL_VENI)
fig_veni = plot_dem_and_stations(*veni_dem_stations)
# fig_veni.savefig('/Users/ldtoney/Downloads/1_veni.png', dpi=400, bbox_inches='tight')

msh_dem_stations = get_dem_and_stations(URL_MSH)
fig_msh = plot_dem_and_stations(*msh_dem_stations)
# fig_msh.savefig('/Users/ldtoney/Downloads/2_msh.png', dpi=400, bbox_inches='tight')

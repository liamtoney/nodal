import cdsapi
import numpy as np
import pandas as pd
import pooch
import xarray as xr

from utils import ERA5_PRESSURE_LEVELS, FULL_REGION, NODAL_WORKING_DIR, get_shots

M_PER_KM = 1000  # [m/km]

# KEY: Select which shot to grab the atmospheric profile for!
SHOT = 'X5'

# Get shot info
df = get_shots()
shot = df.loc[SHOT]
time = pd.Timestamp(shot.time.datetime).round('1h').to_pydatetime()  # Nearest hour!

region = FULL_REGION

c = cdsapi.Client()

params = {
    'product_type': 'reanalysis',
    'format': 'netcdf',
    'variable': [
        'u_component_of_wind',
        'v_component_of_wind',
        'geopotential',
        'temperature',
    ],
    'year': [str(y) for y in np.atleast_1d(time.year)],
    'month': [f'{m:02}' for m in np.atleast_1d(time.month)],
    'day': [f'{d:02}' for d in np.atleast_1d(time.day)],
    'time': [f'{h:02}' for h in np.atleast_1d(time.hour)],
    'pressure_level': [str(l) for l in ERA5_PRESSURE_LEVELS],  # [hPa]
}
if region:  # [xmin, xmax, ymin, ymax]
    params['area'] = [region[3], region[0], region[2], region[1]]

ds = xr.open_dataset(
    pooch.retrieve(
        c.retrieve('reanalysis-era5-pressure-levels', params).location,
        known_hash=None,
    )
).squeeze()

# Select nearest 1D profile
ds_shot = ds.sel(latitude=shot.lat, longitude=shot.lon, method='nearest')
g = 9.81  # [m/s^2]
ds_shot['alt_km'] = (ds_shot.z / g) / M_PER_KM  # [km] Convert to geopotential height

#%% Write .met file

R = 287.058  # [J/kg * K]
density = (ds_shot.level * 100) / (R * ds_shot.t)  # [kg/m^3] Converting pressure to Pa

# TODO: z = 0 in the simulation corresponds to Z_BUFFER below the topographic minimum!
Z_ADJ = -681  # [m] `-profile.min() + Z_BUFFER`

z = ds_shot.alt_km.values + Z_ADJ / M_PER_KM  # [km]
t = ds_shot.t.values  # [K]
u = ds_shot.u.values  # [m/s]
v = ds_shot.v.values  # [m/s]
d = density / 1000  # [g/cm^3] Converting density units here!
p = ds_shot.level  # [mbar]

met_file = NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / f'imush_{SHOT.lower()}.met'
np.savetxt(met_file, np.transpose([z, t, u, v, d, p]), fmt='%.5g')
print(f'Wrote {met_file}')

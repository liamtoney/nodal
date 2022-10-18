from matplotlib import font_manager

from .utils import (
    ERA5_PRESSURE_LEVELS,
    FULL_REGION,
    INNER_RING_REGION,
    MASK_DISTANCE_KM,
    NODAL_WORKING_DIR,
    get_shots,
    get_stations,
    get_waveforms_shot,
    station_map,
)

# TODO: Liam-system-specific!
for font in font_manager.findSystemFonts('/Users/ldtoney/Documents/Helvetica/ttf'):
    font_manager.fontManager.addfont(font)

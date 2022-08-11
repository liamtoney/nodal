#!/usr/bin/env python

"""Downloads StationXML file for iMUSH nodal stations."""

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from utils import NODAL_WORKING_DIR

# Grab "MSH Node Array"
# http://ds.iris.edu/mda/1D/?starttime=2014-01-01T00:00:00&endtime=2014-12-31T23:59:59
print('Downloading coordinates and response information...')
inventory = Client('IRIS').get_stations(
    network='1D',
    starttime=UTCDateTime(2014, 1, 1),
    endtime=UTCDateTime(2014, 12, 31),
    level='response',
    format='xml',
)
print('Done')

# Save as StationXML file
inventory.write(str(NODAL_WORKING_DIR / 'metadata' / '1D.xml'), format='STATIONXML')

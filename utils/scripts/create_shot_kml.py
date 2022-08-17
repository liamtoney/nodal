#!/usr/bin/env python

"""Creates KML file of iMUSH shot locations."""

import simplekml

from utils import NODAL_WORKING_DIR, get_shots

# Load shot metadata
df = get_shots()

# Create KML file
kml = simplekml.Kml()

# Populate file
for _, row in df.iterrows():
    pnt = kml.newpoint(name=row.name)
    pnt.coords = [(row.lon, row.lat)]
    pnt.style.iconstyle.scale = row.weight_lb / 1000  # Scale by shot yield
    pnt.style.iconstyle.icon.href = (
        'http://maps.google.com/mapfiles/kml/shapes/square.png'
    )

# Save KML file
kml.save(NODAL_WORKING_DIR / 'metadata' / 'imush_shots.kml')

#!/usr/bin/env sh

# Download the data archive
curl --remote-name http://lotfi-chaari.net/ens/ImagerieMedicale/TP/data.zip

# Unzip the archive
unzip data.zip

# Delete the archive
rm data.zip

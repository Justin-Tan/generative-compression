#!/bin/bash
# Author: Grace Han
# In place resampling to 512 x 1024 px
# Requires imagemagick on a *nix system
# Modify according to your directory structure

for f in ./**/*.png; do
    convert $f -resize 1024x512 $f
done
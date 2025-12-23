#!/usr/bin/env bash
set -e

gdalbuildvrt ./map/merged.vrt ./map/pred_oBD_30m_v20251219_tile*.tif

gdal_translate ./map/merged.vrt ./map/de_pred_oBD_30m_v20251219.tif \
  -of COG \
  -co COMPRESS=DEFLATE \
  -co PREDICTOR=2 \
  -co BLOCKSIZE=512

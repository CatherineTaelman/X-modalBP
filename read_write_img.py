# ---- This is <read_write_img.py> ----

"""
Read and write image files using gdal.
Author: Johannes Lohse
"""

import os
import sys
from loguru import logger

import numpy as np
from osgeo import gdal

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

def get_all_bands(f, resample=1, asType=np.float32):
  """Read all bands from input image.

  Parameters
  ----------
  f: path to input img file
  resample: resampling interval in both x- and y-direction
  asType: data type

  Returns
  -------
  img: all bands in image file as array
  """

  # open the img file
  img_file = gdal.Open(f)

  # logger
  logger.debug(f'Opened img file: {f}')
  logger.debug(f'Resampling interval set to: {resample}')

  # count the bands
  n_bands = img_file.RasterCount

  # read tif file as image
  if n_bands == 1:
    img = img_file.ReadAsArray()[::resample,::resample].astype(asType)
  else:
    img = img_file.ReadAsArray()[:,::resample,::resample].astype(asType)

  logger.debug(f'Read all bands ({n_bands}) from input file')

  return img

# -------------------------------------------------------------------------- # 
# -------------------------------------------------------------------------- # 

def write_tif(f, X, asType=gdal.GDT_Float32, overwrite=False):
  """Write array (single or multiple bands) to tif file.

  Parameters
  ----------
  f : output tif file
  X : input array
  asType : gdal data type (default: gdal.GDT_Float32)
  overwrite : overwrite if file already exists?

  Returns
  -------
  new_file : return True/False if new tif file has been created
  """

  # check if file already exists
  if os.path.exists(f):
    if overwrite==True:
      logger.debug('File already exists, deleting old file')
      os.remove(f)
    elif overwrite==False:
      logger.info('Output file already exists')
      logger.info('Exiting without writing')
      logger.info('Set overwrite option to True to force')
      new_file = False
      return new_file

  # get dimensions and number of bands
  dims = X.shape
  if np.size(dims) == 2:
    Ny, Nx = X.shape
    n_bands = 1
  elif np.size(dims) == 3:
    n_bands, Ny, Nx = X.shape
    if n_bands > Ny or n_bands > Nx:
      logger.warning('Number of bands is larger than number of pixels')
      logger.warning('Expected shape of input array is (n_bands, Nx, Ny)')
      logger.warning('Exiting without writing')
      new_file = False
      return new_file
  else:
    logger.error(f'Cannot write array with shape {dims} to tif file')
    logger.error('Exiting without writing')
    new_file = False
    return new_file

  # get driver
  output = gdal.GetDriverByName('GTiff').Create(f, Nx, Ny, n_bands, asType)

  # write to file
  if n_bands == 1:
    output.GetRasterBand(1).WriteArray(X)
  elif n_bands > 1:
    for b in np.arange(n_bands):
      output.GetRasterBand(int(b+1)).WriteArray(X[b,:,:])

  output.FlushCache()

  new_file = True

  # logger
  logger.debug('Wrote input array to tif file')

  return new_file

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def write_img(f, X, asType=gdal.GDT_Float32, overwrite=False):
  """Write array (single band) to img file (and hdr header).

  Parameters
  ----------
  f : output img file
  X : input array
  asType : gdal data type (default: gdal.GDT_Float32)
  overwrite : overwrite if file already exists?

  Returns
  -------
  new_file : return True/False if new img file has been created

  Examples
  --------
  write_img('output_file.img', array, asType=gdal.GDT_Float32)
  """

  # check if file already exists
  if os.path.exists(f):
    if overwrite==True:
      logger.debug('File already exists, deleting old file')
      os.remove(f)
      os.remove(os.path.splitext(f)[0]+'.hdr')
    elif overwrite==False:
      logger.info('File already exists')
      logger.info('Exiting without writing')
      logger.info('Set overwrite option to True to force')
      new_file = False
      return new_file

  # get dimensions and number of bands
  dims = X.shape
  if np.size(dims) == 2:
    Ny, Nx = X.shape
    nBands = 1
  else:
    logger.error(f'Cannot write array with shape {dims} to img file')
    logger.error('Exiting without writing')
    new_file = False
    return new_file

  # get driver
  output = gdal.GetDriverByName('Envi').Create(f, Nx, Ny, nBands, asType)

  # write to file
  output.GetRasterBand(1).WriteArray(X)

  output.FlushCache()

  new_File = True

  # logger
  logger.debug('Wrote input array to img file')

  return new_File

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <read_write_img.py> ----

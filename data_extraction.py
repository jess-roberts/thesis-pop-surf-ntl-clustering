import glob
import os
import gdal
import numpy as np
import osr 
import matplotlib.pyplot as plt

class tiffHandle(object):
    def __init__(self,in_filename,out_filename):
        self.input = in_filename
        self.output = out_filename
        self.readReprojRaster()

    def readReprojRaster(self):
        '''
        Read a geotiff in to RAM
        '''
        print("-- Reading in raster --")
        # open a dataset object
        self.ds = gdal.Open(str(self.input))
    
        # read data from geotiff object
        self.nX=self.ds.RasterXSize             # number of pixels in x direction
        self.nY=self.ds.RasterYSize             # number of pixels in y direction
        # geolocation tiepoint
        transform_ds = self.ds.GetGeoTransform()# extract geolocation information
        self.xOrigin=transform_ds[0]       # coordinate of x corner
        self.yOrigin=transform_ds[3]       # coordinate of y corner
        self.pixelWidth=transform_ds[1]    # resolution in x direction
        self.pixelHeight=transform_ds[5]   # resolution in y direction
        # read data. Returns as a 2D numpy array
        self.r_data=self.ds.GetRasterBand(1).ReadAsArray(0,0,self.nX,self.nY)
        self.g_data=self.ds.GetRasterBand(2).ReadAsArray(0,0,self.nX,self.nY)
        self.b_data=self.ds.GetRasterBand(3).ReadAsArray(0,0,self.nX,self.nY)


    def classRaster(self):
        """
            Reclassing non-data as such
        """
        print('-- Reclassing raster --')
        self.r_data_new =self.r_data.copy()
        self.b_data_new =self.b_data.copy()
        self.g_data_new =self.g_data.copy()

        self.r_data_new[(self.r_data == 0) & (self.b_data == 0) & (self.g_data == 0)] = np.nan
        self.b_data_new[(self.r_data == 0) & (self.b_data == 0) & (self.g_data == 0)] = np.nan
        self.g_data_new[(self.r_data == 0) & (self.b_data == 0) & (self.g_data == 0)] = np.nan
    
    def writeTiff(self,epsg=3857):
        """
            Create output raster
        """
        print('-- Creating output raster --')
        # set geolocation information (note geotiffs count down from top edge in Y)
        geotransform = (self.xOrigin, self.pixelWidth, 0, self.yOrigin, 0, self.pixelHeight)

        # load data in to geotiff object
        dst_ds = gdal.GetDriverByName('GTiff').Create(self.output, self.nX, self.nY, 3, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)    # specify coords
        srs = osr.SpatialReference()            # establish encoding
        srs.ImportFromEPSG(epsg)                # set crs
        dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file

        dst_ds.GetRasterBand(1).SetNoDataValue(np.nan)  # set no data value

        dst_ds.GetRasterBand(1).WriteArray(self.r_data_new)  # write image to the raster
        dst_ds.GetRasterBand(2).WriteArray(self.g_data_new)  # write image to the raster
        dst_ds.GetRasterBand(3).WriteArray(self.b_data_new)  # write image to the raster
        
        dst_ds.FlushCache()                     # write to disk
        dst_ds = None
        
        print("Image written to",self.output)

tif = tiffHandle(in_filename='../../RGB_output/malawi_pop_ntl_surf_RGB.tif',out_filename='../../RGB_output/malawi_RGB_data_only.tif')
tif.classRaster()
tif.writeTiff()
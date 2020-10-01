import numpy as np
import gdal
import osr
import matplotlib.pyplot as plt

class rast2array(object):
    def __init__(self,filename):
        self.readRaster(filename)

    def readRaster(self,filename):
        '''
        Read a geotiff in to RAM
        '''
        print("--Reading in raster--")
        # open a dataset object
        ds=gdal.Open(str(filename))

        # read data from geotiff object
        self.nX=ds.RasterXSize             # number of pixels in x direction
        self.nY=ds.RasterYSize             # number of pixels in y direction
        # geolocation tiepoint
        transform_ds = ds.GetGeoTransform()# extract geolocation information
        self.xOrigin=transform_ds[0]       # coordinate of x corner
        self.yOrigin=transform_ds[3]       # coordinate of y corner
        self.pixelWidth=transform_ds[1]    # resolution in x direction
        self.pixelHeight=transform_ds[5]   # resolution in y direction
        # read data. Returns as a 2D numpy array
        self.data=ds.GetRasterBand(1).ReadAsArray(0,0,self.nX,self.nY)
        
        return self

class makeRGB(object):
    def __init__(self,r,g,b,dest):
        self.writeTiff(r,g,b,dest)

    def writeTiff(self,r,g,b,dest):
        """
            Create output raster
        """
        # set geolocation information
        geotransform = (r.xOrigin, r.pixelWidth, 0, r.yOrigin, 0, r.pixelHeight)
        # load data in to geotiff object
        dst_ds = gdal.GetDriverByName('GTiff').Create(dest, r.nX, r.nY, 3, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)    # specify coords
        srs = osr.SpatialReference()            # establish encoding
        srs.ImportFromEPSG(3857)                # encoding the EPSG
        dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
        dst_ds.GetRasterBand(1).SetNoDataValue(np.nan)  # set no data value

        dst_ds.GetRasterBand(1).WriteArray(r.data)  # write red image to the raster
        dst_ds.GetRasterBand(2).WriteArray(g.data)  # write green image to the raster
        dst_ds.GetRasterBand(3).WriteArray(b.data)  # write blue image to the raster
        
        dst_ds.FlushCache()                     # write to disk
        dst_ds = None
        
# Loading in each file for combination
red = rast2array('../../HRSL/malawi_POP.tif')
green = rast2array('../../ISEI/malawi_cci_built_reclass.tif')
blue = rast2array('../../NTL/malawi_NTL_new.tif')

# Getting rid of extraneous night lights and impervious surfaces
print('-- Removing extraneous pixels --')
blue.data[np.where(red.data == 0)] = 0
green.data[np.where(red.data == 0)] = 0

# Start output of the RGB
rgb_dest = '../../RGB_output/malawi_pop_ntl_surf_RGB.tif'
try:    
    rgb_img = makeRGB(r=red,b=blue,g=green,dest=rgb_dest)
    print('RGB image exported to',rgb_dest)
except:
    print('Creation of RGB image failed.')


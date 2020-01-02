import os
import pydicom
from pydicom.tag import Tag
import numpy

PrivateTagData = Tag(0x7fe1, 0x1001)

def HasRawData(ds):
    try:
        # if the private tag is accessible then there is LOGIQ raw data
        _ = ds[PrivateTagData][0]
        return True
    except (IndexError, KeyError, ValueError,pydicom.errors.InvalidDicomError,IOError,TypeError,ValueError,os.error) as err:
        print("Error Not raw data File {0} ".format(err))
        return False

def GetGridSizeID(ds):
    lenSeq = ds[PrivateTagData][0][0x7fe1, 0x1070].VM
    for i in range(0,lenSeq):
        name_ = ds[PrivateTagData][0][0x7fe1, 0x1070][i][0x7fe1, 0x1072].value
        if name_ == 'GridSize':
            return ds[PrivateTagData][0][0x7fe1, 0x1070][i][0x7fe1, 0x1071].value
    return None

def GetGridSize(ds, gridSizeId):
    lenSeq = ds[0x7fe1, 0x1026].VM
    for i in range(0,lenSeq):
        tmp = ds[0x7fe1, 0x1026][i][0x7fe1,0x1048].value
        if isinstance(tmp,(int,float, str)) is True:
            idx = int(tmp)
        elif isinstance(tmp, list) is True:
            idx = int(tmp[0])
        else:
            idx = -1
        if gridSizeId == idx:
            return ds[0x7fe1, 0x1026][i][0x7fe1,0x1086].value
    return None

def ProcessFlow(ds, gridSizeId):
    lenSeq2 = ds[0x7fe1,0x1020].VM
    for j in range(0,lenSeq2):
        name_ = ds[0x7fe1,0x1020][j][0x7fe1,0x1024].value
        if name_ == 'CF':
            CFds = ds[0x7fe1,0x1020][j]
            gridSize = GetGridSize(CFds,gridSizeId)
            numFrames = int(CFds[0x7fe1,0x1036][0][0x7fe1,0x1037].value)
            SWEFrame_ = CFds[0x7fe1,0x1036][0][0x7fe1,0x1061].value
            numpy_format = numpy.dtype('uint16')
            arr = numpy.fromstring(SWEFrame_, numpy_format)
            ndims = gridSize[3]
            samples = gridSize[0]
            beams = gridSize[1] if ndims > 1 else 1
            planes = gridSize[2] if ndims > 2 else 1
            arr = arr[0:samples*beams*planes*numFrames] #seems to be some padding at the end of the data. Shave it off to prevent error in reshaping
            arr = arr.reshape((samples, beams, planes, numFrames), order = 'F')
            return arr
    return None


def SWEFrame(ds, gridSizeId):
    # get the length of the sequence
    lenSeq = ds.VM
    for i in range(0,lenSeq):
        flowDS = ds[i]
        value = flowDS[0x7fe1,0x1012].value
        if value == '2DFlow':
            return ProcessFlow(flowDS, gridSizeId)
    return None


def GetSWEFrame(filename):
    dataset = pydicom.read_file(filename)
    hasRawData = HasRawData(dataset)
    if hasRawData is False:
        return None # failure no raw data in file

    gridSizeId = GetGridSizeID(dataset)
    if gridSizeId is None:
        return None # failure no grid size something wrong with the file

    # point to the private data
    privDS = dataset[PrivateTagData][0][0x7fe1, 0x1010]

    return SWEFrame(privDS, gridSizeId)

if __name__ == "__main__":  

    workDir = 'C:\\dev\\Partners\\GEMS_IMG'
    for root, dirs, files in os.walk(workDir):
        for name in files:
            fullname = os.path.join(root,name)
            try:
                frame = GetSWEFrame(fullname)
                if frame is not None:
                    print('{0} has SWE frame '.format(fullname))
            except:
                print('unhandled exception')
                continue
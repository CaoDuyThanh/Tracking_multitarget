########################################################################################################################
#                                                                                                                      #
#                          TOWN CENTER DATASET                                                                         #
#                                                                                                                      #
########################################################################################################################
import numpy
import os
from DataHelper import DatasetHelper
from FileHelper import *

class TCDataHelper(DatasetHelper):

    def __init__(self,
                 datasetPath = None):
        DatasetHelper.__init__(self)

        # Check parameters
        check_not_none(datasetPath, 'datasetPath'); check_path_exist(datasetPath)

        # Set parameters
        self.DatasetPath  = datasetPath

        # Config train | test dataset
        self.TrainPath    = self.DatasetPath + 'train/'
        self.TestPath     = self.DatasetPath + 'test/'
        self.TrainFolders = get_all_sub_folders_path(self.TrainPath)
        self.TestFolders  = get_all_sub_folders_path(self.TestPath)

        # Load data
        self.load_train_file()
        self.load_test_file()

    # ------------------------  CHECK BBOX IN RANGE OF IMAGE  ---------------------------------------------
    def checkBBox(self, topLeftX, topLeftY, width, height, realWidth, realHeight):
        bottomRightX = topLeftX + width
        bottomRightY = topLeftY + height

        topLeftX     = max(0, topLeftX)
        topLeftY     = max(0, topLeftY)
        bottomRightX = min(bottomRightX, realWidth)
        bottomRightY = min(bottomRightY, realHeight)

        cx = (topLeftX + bottomRightX) * 1. / 2
        cy = (topLeftY + bottomRightY) * 1. / 2
        width    = bottomRightX - topLeftX
        height   = bottomRightY - topLeftY

        cx     = cx * 1. / realWidth
        cy     = cy * 1. / realHeight
        width  = width * 1. / realWidth
        height = height * 1. / realHeight

        return [cx, cy, width, height]

    # ------------------------  LOAD TRAIN | VALID | TEST FILES--------------------------------------------
    def getDataFromOneFolder(self, path):
        # Check folder
        check_path_exist(path)

        # Config img1 folder exist
        img1Folder = path + 'img1/';      check_path_exist(img1Folder)

        # Create empty dictionary stored all data
        Data  = dict()

        # Get image info
        seqInfoPath = path + 'seqinfo.ini'
        seqInfo     = read_file_ini(seqInfoPath)
        imageInfo   = dict()
        imageInfo['imagewidth'] = float(seqInfo['Sequence']['imWidth'])
        imageInfo['imageheight'] = float(seqInfo['Sequence']['imHeight'])
        Data['imageinfo'] = imageInfo

        # Read det.txt
        detFolder  = path + 'det/';
        if check_path_exist(detFolder, _throw_error= False):
            Frames      = dict()
            FramesPath  = dict()
            ObjectId    = dict()
            detFilePath = detFolder + 'det.txt';    check_file_exist(detFilePath)
            allDets     = read_file(detFilePath)
            for det in allDets:
                data = det.split(',')

                frameId  = int(data[0])  # Which frame object appears
                objectId = float(data[1])  # Number identifies that object as belonging to a tragectory by unique ID
                topLeftX = float(data[2])  # Topleft corner of bounding box (x)
                topLeftY = float(data[3])  # Topleft corner of bounding box (y)
                width    = float(data[4])  # Width of the bounding box
                height   = float(data[5])  # Height of the bounding box
                isIgnore = float(data[6])  # Flag whether this particular instance is ignored in the evaluation
                type     = float(data[7])  # Identify the type of object
                                            #     Label                ID
                                            # Pedestrian                1
                                            # Person on vehicle         2
                                            # Car                       3
                                            # Bicycle                   4
                                            # Motorbike                 5
                                            # Non motorized vehicle     6
                                            # Static person             7
                                            # Distrator                 8
                                            # Occluder                  9
                                            # Occluder on the ground   10
                                            # Occluder full            11
                                            # Reflection               12
                imPath   = os.path.join(img1Folder, '%06d.jpg' % (frameId))

                if frameId not in Frames:
                    Frames[frameId] = []
                cx, cy, width, height = self.checkBBox(topLeftX, topLeftY, width, height, imageInfo['imagewidth'], imageInfo['imageheight'])
                Frames[frameId].append([cx, cy, width, height, isIgnore, type, imPath])

                if objectId not in ObjectId:
                    ObjectId[objectId] = frameId

                if frameId not in FramesPath:
                    FramesPath[frameId] = os.path.abspath(os.path.join(img1Folder, '{0:06}'.format(frameId) + '.jpg'))

            DetData = dict()
            DetData['frames']     = Frames
            DetData['objectid']   = ObjectId
            DetData['framespath'] = FramesPath
            Data['det'] = DetData

        # Read gt.txt
        gtFolder = path + 'gt/';
        if check_path_exist(gtFolder, _throw_error= False):
            Frames     = dict()
            FramesPath = dict()
            ObjectId   = dict()
            gtFilePath  = gtFolder + 'gt.txt';      check_file_exist(gtFilePath)
            allGts = read_file(gtFilePath)
            for gt in allGts:
                data = gt.split(',')

                frameId  = int(data[0])      # Which frame object appears
                objectId = int(data[1])      # Number identifies that object as belonging to a tragectory by unique ID
                topLeftX = int(data[2])      # Topleft corner of bounding box (x)
                topLeftY = int(data[3])      # Topleft corner of bounding box (y)
                width    = int(data[4])      # Width of the bounding box
                height   = int(data[5])      # Height of the bounding box
                isIgnore = int(data[6])      # Flag whether this particular instance is ignored in the evaluation
                type     = int(data[7])      # Identify the type of object
                occluder = float(data[8])    # Occluder of object
                                                #     Label                ID
                                                # Pedestrian                1
                                                # Person on vehicle         2
                                                # Car                       3
                                                # Bicycle                   4
                                                # Motorbike                 5
                                                # Non motorized vehicle     6
                                                # Static person             7
                                                # Distrator                 8
                                                # Occluder                  9
                                                # Occluder on the ground   10
                                                # Occluder full            11
                                                # Reflection               12
                imPath    = os.path.join(img1Folder, '%06d.jpg' % (frameId))

                if frameId not in Frames:
                    Frames[frameId] = dict()
                cx, cy, width, height = self.checkBBox(topLeftX, topLeftY, width, height, imageInfo['imagewidth'], imageInfo['imageheight'])
                Frames[frameId][objectId] = [frameId, cx, cy, width, height, isIgnore, type, occluder, imPath]

                if objectId not in ObjectId:
                    ObjectId[objectId] = frameId

                if frameId not in FramesPath:
                    FramesPath[frameId] = os.path.abspath(os.path.join(img1Folder, '{0:06}'.format(frameId) + '.jpg'))

            GtData = dict()
            GtData['frames']     = Frames
            GtData['objectid']   = ObjectId
            GtData['framespath'] = FramesPath
            Data['gt'] = GtData

        return Data

    def load_train_file(self):
        self.TrainData        = dict()
        for trainFolder in self.TrainFolders:
            folderName = trainFolder.split('/')[-2]
            self.TrainData[folderName] = self.getDataFromOneFolder(trainFolder)

    def load_test_file(self):
        self.TestData        = dict()
        for testFolder in self.TestFolders:
            folderName = testFolder.split('/')[-2]
            self.TestData[folderName] = self.getDataFromOneFolder(testFolder)

    # -----------------------------------------------------------------------------------------------------

    def GetFramesPath(self,
                      folderName,
                      startFrame = 1,           # None = Start from the first frame
                      endFrame   = 10000000     # None = To the end of frame
                      ):
        frameId = startFrame
        framesPath = []
        while (frameId < endFrame):
            if frameId not in self.TrainData[folderName]['gt']['framespath']:
                break;
            framesPath.append(self.TrainData[folderName]['gt']['framespath'][frameId])
            frameId += 1

        return framesPath

    def GetAllObjectIds(self):
        if self.data_opts['data_phase'] == 'train':
            folderName   = self.data_opts['data_folder_name']
            folderType   = self.data_opts['data_folder_type']
            data         = self.TrainData[folderName][folderType]
            allObjectIds = [objectId for objectId in data['objectid']]
            return allObjectIds

        if self.data_opts['data_phase'] == 'test':
            assert 'Do not support get AllObjectIds from test'

    def GetRandomBbox(self):
        if self.data_opts['data_phase'] == 'train':
            assert 'Do not support get random object from train'

        if self.data_opts['data_phase'] == 'test':
            folderName   = self.data_opts['data_folder_name']
            folderType   = self.data_opts['data_folder_type']
            data         = self.TestData[folderName][folderType]
            firstFrames  = data['frames'][1]
            ranObject    = firstFrames[1]
            return data['framespath'], ranObject


    def get_all_folder_names(self):
        if self.data_opts['data_phase'] == 'train':
            allFolderNames = [folderName for folderName in self.TrainData]
            return allFolderNames

        if self.data_opts['data_phase'] == 'test':
            allFolderNames = [folderName for folderName in self.TestData]
            return allFolderNames

    def GetSequenceBy(self,
                      occluderThres = 0.5):
        dataPhase  = self.data_opts['data_phase']
        folderName = self.data_opts['data_folder_name']
        folderType = self.data_opts['data_folder_type']
        objectId   = self.data_opts['data_object_id']

        if folderType == 'det':
            assert 'Get sequence data must in gt folder'

        if dataPhase == 'train':
            data       = self.TrainData[folderName]
            ObjectId   = data[folderType]['objectid']
            frameStart = ObjectId[objectId]
            Frames     = data[folderType]['frames']
            imsPath    = []
            bbox       = []
            while frameStart < Frames.__len__():
                currentFrame = Frames[frameStart]
                if objectId in currentFrame:
                    if currentFrame[objectId][7] >= occluderThres:
                        imsPath.append(currentFrame[objectId][-1])
                        bbox.append(currentFrame[objectId][1:5])
                else:
                    break;
                frameStart += 1

            return imsPath, bbox

        if self.data_opts['data_phase'] == 'test':
            assert 'Do not support get Sequence from test'


    def next_train_batch(self): raise NotImplementedError

    def next_valid_batch(self): raise NotImplementedError

    def next_test_batch(self, _batch_size): raise NotImplementedError
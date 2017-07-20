import numpy
import os
from DataHelper import DatasetHelper
from FileHelper import *

class MOTDataHelper(DatasetHelper):

    def __init__(self,
                 _dataset_path = None
                 ):
        DatasetHelper.__init__(self)

        # Check parameters
        check_not_none(_dataset_path, 'datasetPath'); check_path_exist(_dataset_path)

        # Set parameters
        self.dataset_path  = _dataset_path

        # Config train | test dataset
        self.train_path    = self.dataset_path + 'train/'
        self.test_path     = self.dataset_path + 'test/'
        self.train_folders = get_all_sub_folders_path(self.train_path)
        self.test_folders  = get_all_sub_folders_path(self.test_path)

        # Load data
        self.load_train_file()
        self.load_test_file()

    # ------------------------  CHECK BBOX IN RANGE OF IMAGE  ---------------------------------------------
    def _check_bbox(self, _top_left_x, _top_left_y, _width, _height, _real_width, _real_height):
        _bottom_right_x = _top_left_x + _width
        _bottom_right_y = _top_left_y + _height

        _top_left_x     = max(0, _top_left_x)
        _top_left_y     = max(0, _top_left_y)
        _bottom_right_x = min(_bottom_right_x, _real_width)
        _bottom_right_y = min(_bottom_right_y, _real_height)

        _cx = (_top_left_x + _bottom_right_x) * 1. / 2
        _cy = (_top_left_y + _bottom_right_y) * 1. / 2
        _width    = _bottom_right_x - _top_left_x
        _height   = _bottom_right_y - _top_left_y

        cx     = _cx * 1. / _real_width + 0.000001
        cy     = _cy * 1. / _real_height + 0.000001
        width  = _width * 1. / _real_width
        height = _height * 1. / _real_height

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
                imPath   = os.path.join(img1Folder, '300x300/%06d.jpg' % (frameId))         # Change this

                if frameId not in Frames:
                    Frames[frameId] = []
                cx, cy, width, height = self._check_bbox(topLeftX, topLeftY, width, height, imageInfo['imagewidth'], imageInfo['imageheight'])
                Frames[frameId].append([cx, cy, width, height, isIgnore, type, imPath])

                if objectId not in ObjectId:
                    ObjectId[objectId] = frameId

                if frameId not in FramesPath:
                    FramesPath[frameId] = os.path.abspath(os.path.join(img1Folder + '300x300/', '{0:06}'.format(frameId) + '.jpg'))

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
                imPath    = os.path.join(img1Folder, '300x300/%06d.jpg' % (frameId))            # Change this

                if frameId not in Frames:
                    Frames[frameId] = dict()
                cx, cy, width, height = self._check_bbox(topLeftX, topLeftY, width, height, imageInfo['imagewidth'], imageInfo['imageheight'])
                Frames[frameId][objectId] = [frameId, cx, cy, width, height, isIgnore, type, occluder, imPath]

                if objectId not in ObjectId:
                    ObjectId[objectId] = frameId

                if frameId not in FramesPath:
                    FramesPath[frameId] = os.path.abspath(os.path.join(img1Folder + '300x300/   ', '{0:06}'.format(frameId) + '.jpg'))

            GtData = dict()
            GtData['frames']     = Frames
            GtData['objectid']   = ObjectId
            GtData['framespath'] = FramesPath
            Data['gt'] = GtData

        return Data

    def load_train_file(self):
        self.TrainData        = dict()
        for trainFolder in self.train_folders:
            folderName = trainFolder.split('/')[-2]
            self.TrainData[folderName] = self.getDataFromOneFolder(trainFolder)

    def load_test_file(self):
        self.TestData        = dict()
        for testFolder in self.test_folders:
            folderName = testFolder.split('/')[-2]
            self.TestData[folderName] = self.getDataFromOneFolder(testFolder)

    # -----------------------------------------------------------------------------------------------------
    def get_frames_path(self,
                        folderName,
                        startFrame = 1,  # None = Start from the first frame
                      endFrame   = 10000000  # None = To the end of frame
                        ):
        frameId = startFrame
        framesPath = []
        while (frameId < endFrame):
            if frameId not in self.TrainData[folderName]['gt']['framespath']:
                break;
            framesPath.append(self.TrainData[folderName]['gt']['framespath'][frameId])
            frameId += 1

        return framesPath

    def get_all_object_ids(self):
        if self.data_opts['data_phase'] == 'train':
            folderName   = self.data_opts['data_folder_name']
            folderType   = self.data_opts['data_folder_type']
            data         = self.TrainData[folderName][folderType]
            allObjectIds = [objectId for objectId in data['objectid']]
            return allObjectIds

        if self.data_opts['data_phase'] == 'test':
            assert 'Do not support get AllObjectIds from test'

    def get_random_bbox(self):
        if self.data_opts['data_phase'] == 'train':
            folderName = self.data_opts['data_folder_name']
            folderType = self.data_opts['data_folder_type']
            data = self.TrainData[folderName][folderType]
            firstFrames = data['frames'][1]
            ranObject = firstFrames[1]
            return data['framespath'], ranObject

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

    def get_object_ids_by_frame(self,
                                _frame_id):
        _data_phase  = self.data_opts['data_phase']
        _folder_name = self.data_opts['data_folder_name']
        _folder_type = self.data_opts['data_folder_type']

        if _data_phase == 'det':
            assert 'Get object ids by frame must in gt folder'

        if _data_phase == 'train':
            _data           = self.TrainData[_folder_name]
            _frames         = _data[_folder_type]['frames']
            _selected_frame = _frames[_frame_id]

            all_bboxs      = []
            all_object_ids = []
            for (_object_id, _object_data) in _selected_frame.items():
                all_bboxs.append(_object_data[1 : 5])
                all_object_ids.append(_object_id)

            all_bboxs      = numpy.asarray(all_bboxs)
            all_object_ids = numpy.asarray(all_object_ids)
            return all_bboxs, all_object_ids

    def get_sequence_by(self,
                        occluderThres = 0.5):
        dataPhase  = self.data_opts['data_phase']
        folderName = self.data_opts['data_folder_name']
        folderType = self.data_opts['data_folder_type']
        objectId   = self.data_opts['data_object_id']

        if dataPhase == 'det':
            assert 'Get sequence data must in gt folder'

        if dataPhase == 'train':
            data        = self.TrainData[folderName]
            object_id   = data[folderType]['objectid']
            frame_start = object_id[objectId]
            frames      = data[folderType]['frames']
            ims_path    = []
            bbox        = []
            while frame_start < max(frames):
                currentFrame = frames[frame_start]
                if objectId in currentFrame:
                    if currentFrame[objectId][7] >= occluderThres:
                        ims_path.append(currentFrame[objectId][-1])
                        bbox.append(currentFrame[objectId][1:5])
                frame_start += 1

            return ims_path, bbox, frame_start

        if self.data_opts['data_phase'] == 'test':
            assert 'Do not support get Sequence from test'

    def next_train_batch(self): raise NotImplementedError

    def next_valid_batch(self): raise NotImplementedError

    def next_test_batch(self, _batch_size): raise NotImplementedError
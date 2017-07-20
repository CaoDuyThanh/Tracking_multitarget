import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy
from matplotlib.gridspec import GridSpec

class SubplotManager():
    def __init__(self,
                 _fig):
        self.figure   = _fig;       self.figure.axis('off')
        self.fig_data = None
        self.rects    = []
        self.old_res  = 1
        self.img      = None

        self.fixed_color_bboxs = numpy.zeros((100, 3))
        for _color_id in range(100):
            self.fixed_color_bboxs[_color_id,] = numpy.random.rand(3, 1).T

    def update_img(self,
                   _img):
        self.img = _img
        _h, _w, _ = _img.shape
        if self.fig_data == None:
            self.fig_data = self.figure.imshow(_img)
            # self.old_res = _w * 1. / _h
        else:
            self.fig_data.set_data(_img)
            # self.figure.set_aspect(_h * 1. / _w / self.old_res)

    def clear(self):
        if self.fig_data:
            self.fig_data.remove()
            self.fig_data = None

    def _remove_rects(self):
        if len(self.rects) != 0:
            for _rect in self.rects:
                _rect.remove()
            self.rects = []

    def update_bboxs(self,
                     _id_bboxs,
                     _bboxs):
        self._remove_rects()
        _h, _w, _ = self.img.shape
        for _id, _box in enumerate(_bboxs):
            if int(_id_bboxs[_id]) > 0:
                _color = self.fixed_color_bboxs[int(_id_bboxs[_id]) - 1]
                _rect  = patches.Rectangle(((_box[0] - _box[2] / 2) * _w,
                                            (_box[1] - _box[3] / 2) * _h),
                                             _box[2] * _w,
                                             _box[3] * _h,
                                           linewidth = 3,
                                           edgecolor = _color,
                                           facecolor = 'none')
                self.rects.append(_rect)
                # Add the patch to the Axes
                self.figure.add_patch(_rect)

class PlotManager():
    TOTAL_ROWS      = 6
    TOTAL_COLS      = 8
    MAIN_PLOT_START = 6

    def __init__(self):
        # Default settings
        plt.ion()

        # Create figure and subplots
        self.main_figure = plt.figure(figsize=(12, 6));    plt.axis('off');     self.main_figure.tight_layout()
        self.grid_spec   = GridSpec(self.TOTAL_ROWS, self.TOTAL_COLS)

        _main_plot_fig   = self.main_figure.add_subplot(self.grid_spec[0 : self.MAIN_PLOT_START,
                                                                       0 : self.MAIN_PLOT_START])
        self.main_plot   = SubplotManager(_main_plot_fig)
        self.rects       = []
        self.mini_plots  = []
        for _plot_row_id in range(self.TOTAL_ROWS):
            for _plot_col_id in range(self.TOTAL_COLS):
                if _plot_col_id < self.MAIN_PLOT_START and \
                   _plot_row_id < self.MAIN_PLOT_START:
                    continue
                _sub_plot = self.main_figure.add_subplot(self.grid_spec[_plot_row_id, _plot_col_id])
                self.mini_plots.append(SubplotManager(_sub_plot))

    def update_main_plot(self,
                         _img):
        self.main_plot.update_img(_img)

    def draw_bboxs(self,
                   _id_bboxs,
                   _bboxs):
        self.main_plot.update_bboxs(_id_bboxs, _bboxs)

    def draw_mini_bboxs(self,
                        _img,
                        _id_bboxs,
                        _bboxs):
        _check = numpy.zeros((len(self.mini_plots)))
        _h, _w, _  = _img.shape
        for _id, _id_bbox in enumerate(_id_bboxs):
            if _id_bbox > 0:
                _bbox   = _bboxs[_id]
                _cx     = _bbox[0]
                _cy     = _bbox[1]
                _width  = _bbox[2]
                _height = _bbox[3]
                _xmin   = int((_cx - _width / 2) * _w)
                _xmax   = int((_cx + _width / 2) * _w)
                _ymin   = int((_cy - _height / 2) * _h)
                _ymax   = int((_cy + _height / 2) * _h)
                _sub_image = _img[_ymin : _ymax, _xmin : _xmax, ]

                if int(_id_bbox) - 1 < len(self.mini_plots):
                    self.mini_plots[int(_id_bbox) - 1].update_img(_sub_image)
                    _check[int(_id_bbox) - 1] = 1

        for _id in range(len(_check)):
            if _check[_id] == 0:
                self.mini_plots[_id].clear()


    def refresh(self):
        plt.show()
        plt.pause(0.001)


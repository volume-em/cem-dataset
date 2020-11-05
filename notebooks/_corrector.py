"""
Copied with modification from https://github.com/fastai/fastai1/blob/6a5102ef7bdefa9058d0481ab311f48b21cbc6fc/fastai/widgets/image_cleaner.py

Modifications:
--------------
1. Added new Dataset class
2. Changed widget to show a grid with multiple rows

"""

import cv2
import numpy as np
from abc import ABC
from itertools import chain, islice
from math import ceil
from ipywidgets import widgets, Layout
from IPython.display import clear_output, display
from dataclasses import dataclass
from collections import Mapping

from skimage.io import imread

class Dataset:
    def __init__(self, fnames, labels, num_images=None, eval_label=None):
        self.fnames = fnames
        self.labels = labels
        self.eval_label = eval_label
        self.num_images = num_images
        
        if eval_label is not None:
            self.label_indices = np.where(self.labels == eval_label)[0]
            self.fnames = self.fnames[self.label_indices]
            self.labels = self.labels[self.label_indices]
            
        else:
            self.label_indices = np.array(list(range(len(self.fnames))))
        
        if num_images is not None:
            self.permutation = np.random.randint(0, len(self.fnames), num_images)
            self.fnames = self.fnames[self.permutation]
            self.labels = self.labels[self.permutation]
        else:
            self.permutation = np.array(list(range(len(self.fnames))))
        
    def __len__(self):
        return len(self.fnames)
        
    def __getitem__(self, idx):
        return cv2.imencode('.jpg', imread(self.fnames[idx]))[1].tobytes()

@dataclass
class ImgData:
    jpg_blob: bytes
    label: str
    payload: Mapping

class BasicImageWidget(ABC):
    def __init__(self, dataset, batch_size=5, classes=None, rows=1):
        super().__init__()
        self.dataset = dataset
        self.labels = dataset.labels
        self.batch_size = batch_size
        self.classes = classes
        self.rows = rows
        assert(self.batch_size % rows == 0), "Batch size must be divisible by rows!"
        self.cols = batch_size // self.rows

        self._all_images = self.create_image_list()

    @staticmethod
    def make_img_widget(img, layout=Layout(height='150px', width='150px'), format='jpg'):
        "Returns an image widget for specified file name `img`."
        return widgets.Image(value=img, format=format, layout=layout)

    @staticmethod
    def make_button_widget(label, handler, img_idx=None, style=None, layout=Layout(width='auto')):
        "Return a Button widget with specified `handler`."
        btn = widgets.Button(description=label, layout=layout)
        btn.on_click(handler)
        if style is not None: 
            btn.button_style = style
        if img_idx is not None: 
            btn.img_idx = img_idx
        return btn

    @staticmethod
    def make_dropdown_widget(options, value, handler, img_idx=None, description='', layout=Layout(width='auto')):
        "Return a Dropdown widget with specified `handler`."
        dd = widgets.Dropdown(description=description, options=options, value=value, layout=layout)
        dd.observe(handler, names='value')
        if img_idx is not None: 
            dd.img_idx = img_idx
        return dd

    @staticmethod
    def make_horizontal_box(children, layout=Layout()):
        "Make a grid box with `children` and `layout`."
        return widgets.GridBox(children, layout=layout)
    
    @staticmethod
    def make_vertical_box(children, layout=Layout(width='auto', height='300px', overflow_x="hidden")):
        "Make a vertical box with `children` and `layout`."
        return widgets.VBox(children, layout=layout)

    def create_image_list(self):
        "Create a list of images, filenames and labels but first removing files that are not supposed to be displayed."
        items = self.dataset
        idxs = range(len(items))
        for i in idxs: 
            yield ImgData(self.dataset[i], self._get_label(i), self.make_payload(i))

    def _get_label(self, idx):
        "Returns a label for an image with the given `idx`."
        return self.labels[idx]

    def make_payload(self, idx:int):
        "Override in a subclass to associate an image with the given `idx` with a custom payload."
        pass

    def _get_change_payload(self, change_owner):
        """
        Call in widget's on change handler to retrieve the payload.
        Assumes the widget was created by a factory method taking `img_idx` parameter.
        """
        return self._batch_payloads[change_owner.img_idx]

    def next_batch(self, _=None):
        "Fetches a next batch of images for rendering."
        if self.before_next_batch and hasattr(self, '_batch_payloads'): 
            self.before_next_batch(self._batch_payloads)
        batch = tuple(islice(self._all_images, self.batch_size))
        self._batch_payloads = tuple(b.payload for b in batch)
        self.render(batch)

    def render(self, batch):
        "Override in a subclass to render the widgets for a batch of images."
        pass

class PredictionsCorrector(BasicImageWidget):
    "Displays images for manual inspection and relabelling."
    def __init__(self, dataset, classes, batch_size=5, rows=1):
        super().__init__(dataset, batch_size, classes=classes, rows=rows)
        self.corrections = {}
        self.before_next_batch = None
        self.next_batch()

    def show_corrections(self, ncols, **fig_kw):
        "Shows a grid of images whose predictions have been corrected."
        nrows = ceil(len(self.corrections) / ncols)
        fig, axs = plt.subplots(nrows, ncols, **fig_kw)
        axs, extra_axs = np.split(axs.flatten(), (len(self.corrections),))

        for (idx, new), ax in zip(sorted(self.corrections.items()), axs):
            old = self._get_label(idx)
            self.dataset[idx].show(ax=ax, title=f'{idx}: {old} -> {new}')

        for ax in extra_axs:
            ax.axis('off')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

    def corrected_labels(self):
        "Returns labels for the entire test set with corrections applied."
        corrected = list(self.labels)
        for i, l in self.corrections.items():
            index = self.dataset.label_indices[self.dataset.permutation[i]]
            corrected[index] = l
        return corrected

    def make_payload(self, idx): 
        return {'idx': idx}

    def render(self, batch):
        clear_output()
        if not batch:
            return display('No images to show :)')
        else:
            grid_template = ('1fr ' * self.cols)[:-1]
            display(self.make_horizontal_box(self.get_widgets(batch), widgets.Layout(grid_template_columns=grid_template)))
            display(self.make_button_widget('Next Batch', handler=self.next_batch, style='primary'))

    def get_widgets(self, batch):
        widgets = []
        for i, img in enumerate(batch):
            img_widget = self.make_img_widget(img.jpg_blob)
            dropdown = self.make_dropdown_widget(options=self.classes, value=img.label,
                                                 handler=self.relabel, img_idx=i)
            widgets.append(self.make_vertical_box((img_widget, dropdown)))
        return widgets

    def relabel(self, change):
        self.corrections[self._get_change_payload(change.owner)['idx']] = change.new
import os
import xml.etree.ElementTree as ET

import numpy as np
import os
import cv2
import sys
from .util import read_image

VOC_BBOX_LABEL_NAMES = (
	'zebra',
	'zebra_back')
class ZebraBboxDataset:
	"""Bounding box dataset for Zebra Back dataset.

	

	The index corresponds to each image.

	When queried by an index, if :obj:`return_difficult == False`,
	this dataset returns a corresponding
	:obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
	This is the default behaviour.
	If :obj:`return_difficult == True`, this dataset returns corresponding
	:obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
	that indicates whether bounding boxes are labeled as difficult or not.

	The bounding boxes are packed into a two dimensional tensor of shape
	:math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
	the image. The second axis represents attributes of the bounding box.
	They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
	four attributes are coordinates of the top left and the bottom right
	vertices.

	The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
	:math:`R` is the number of bounding boxes in the image.
	The class name of the label :math:`l` is :math:`l` th element of
	:obj:`VOC_BBOX_LABEL_NAMES`.

	The array :obj:`difficult` is a one dimensional boolean array of shape
	:math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
	If :obj:`use_difficult` is :obj:`False`, this array is
	a boolean array with all :obj:`False`.

	The type of the image, the bounding boxes and the labels are as follows.

	* :obj:`img.dtype == numpy.float32`
	* :obj:`bbox.dtype == numpy.float32`
	* :obj:`label.dtype == numpy.int32`
	* :obj:`difficult.dtype == numpy.bool`

	Args:
		data_dir (string): Path to the root of the training data. 
			i.e. "/data/image/voc/VOCdevkit/VOC2007/"
		split ({'train', 'val', 'trainval', 'test'}): Select a split of the
			dataset. :obj:`test` split is only available for
			2007 dataset.
		year ({'2007', '2012'}): Use a dataset prepared for a challenge
			held in :obj:`year`.
		use_difficult (bool): If :obj:`True`, use images that are labeled as
			difficult in the original annotation.
		return_difficult (bool): If :obj:`True`, this dataset returns
			a boolean array
			that indicates whether bounding boxes are labeled as difficult
			or not. The default value is :obj:`False`.

	"""

	def __init__(self, data_dir, split='trainval',
				 use_difficult=False, return_difficult=False,
				 ):

		# if split not in ['train', 'trainval', 'val']:
		#     if not (split == 'test' and year == '2007'):
		#         warnings.warn(
		#             'please pick split from \'train\', \'trainval\', \'val\''
		#             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
		#             ' in addition to the above mentioned splits.'
		#         )
		if split == 'trainval':
			id_list_file = '/user/work/gh18931/diss/datasets/task_simple-frcnn-data/annotation.txt'
		else:
			id_list_file = '/user/work/gh18931/diss/datasets/task_simple-frcnn-data/test_annotation.txt'
		#id_list_file = os.path.join( data_dir, 'ImageSets/Main/{0}.txt'.format(split))
		self.imgs, self.classes_count, self.class_mapping = self.get_data(id_list_file)
		#self.imges = list(filepath, width, height,class, list(bboxes))
		self.ids =[]
		for img in self.imgs:
			
			if img["filepath"] not in self.ids:
				self.ids.append(img["filepath"])
		self.use_difficult = use_difficult
		self.return_difficult = return_difficult
		self.label_names = VOC_BBOX_LABEL_NAMES
		
	def __len__(self):
		return len(self.ids)

	def get_example(self, i):
		"""Returns the i-th example.

		Returns a color image and bounding boxes. The image is in CHW format.
		The returned image is RGB.

		Args:
			i (int): The index of the example.

		Returns:
			tuple of an image and bounding boxes

		"""
		id = self.ids[i]
		
		bbox = list()
		label = list()
		difficult = list()
		for img in self.imgs:

			if img["filepath"] is id:

				for bb in img["bboxes"]:

					bbox.append([bb["y1"],bb["x1"],bb["y2"],bb["x2"]])
					difficult.append(0)
					name = bb["class"]
					label.append(int(name)-1)
		bbox = np.stack(bbox).astype(np.float32)
		label = np.stack(label).astype(np.int32)
		
		difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

		# Load a image
		img_file = os.path.join(id)
		img = read_image(img_file, color=True)
		
		# if self.return_difficult:
		#     return img, bbox, label, difficult
		return img, bbox, label, difficult

	__getitem__ = get_example

#----------------------- Helper function----------------
	def get_data(self,input_path):
		"""Parse the data from annotation file

		Args:
			input_path: annotation file path

		Returns:
			all_data: list(filepath, width, height,class, list(bboxes))
			classes_count: dict{key:class_name, value:count_num}
				e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
			class_mapping: dict{key:class_name, value: idx}
				e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
		"""
		found_bg = False
		all_imgs = {}

		classes_count = {}

		class_mapping = {}

		visualise = True

		i = 1

		with open(input_path,'r') as f:

			print('Parsing annotation files')

			for line in f:

				# Print process
				sys.stdout.write('\r'+'idx=' + str(i))
				i += 1

				line_split = line.strip().split(',')

				# Make sure the info saved in annotation file matching the format (path_filename, x1, y1, x2, y2, class_name)
				# Note:
				#	One path_filename might has several classes (class_name)
				#	x1, y1, x2, y2 are the pixel value of the origial image, not the ratio value
				#	(x1, y1) top left coordinates; (x2, y2) bottom right coordinates
				#   x1,y1-------------------
				#	|						|
				#	|						|
				#	|						|
				#	|						|
				#	---------------------x2,y2

				(filename,x1,y1,x2,y2,class_name) = line_split

				if class_name not in classes_count:
					classes_count[class_name] = 1
				else:
					classes_count[class_name] += 1

				if class_name not in class_mapping:
					if class_name == 'bg' and found_bg == False:
						print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
						found_bg = True
					class_mapping[class_name] = len(class_mapping)

				if filename not in all_imgs:
					all_imgs[filename] = {}
					
					img = cv2.imread(filename)
					(rows,cols) = img.shape[:2]
					all_imgs[filename]['filepath'] = filename
					all_imgs[filename]['width'] = cols
					all_imgs[filename]['height'] = rows
				
					all_imgs[filename]['bboxes'] = []
					# if np.random.randint(0,6) > 0:
					# 	all_imgs[filename]['imageset'] = 'trainval'
					# else:
					# 	all_imgs[filename]['imageset'] = 'test'

				all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


			all_data = []
			for key in all_imgs:
				all_data.append(all_imgs[key])

			# make sure the bg class is last in the list
			if found_bg:
				if class_mapping['bg'] != len(class_mapping) - 1:
					key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
					val_to_switch = class_mapping['bg']
					class_mapping['bg'] = len(class_mapping) - 1
					class_mapping[key_to_switch] = val_to_switch

			return all_data, classes_count, class_mapping




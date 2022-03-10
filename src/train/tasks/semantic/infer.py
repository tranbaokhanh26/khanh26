#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
#import in infer.py
import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import time
import collections
import copy
import cv2
import numpy as np
import torch.nn.functional as F
import math
from scipy import signal
from PIL import Image
import rospy
from lidar_bonnetal.msg import dulieu

from torch.utils.data import Dataset

class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']
  def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.reset()

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)       # [H,W] mask

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    # print(points[15])
    remissions = scan[:, 3]  # get remission
    self.set_points(points, remissions)

  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz
    # print(points)
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      self.do_range_projection()

  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)
    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]
    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    self.unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx > 0).astype(np.int32)


class SemLaserScan(LaserScan):
  """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
  EXTENSIONS_LABEL = ['.label']

  def __init__(self,  sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, max_classes=300):
    super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
    self.reset()

    # make semantic colors
    if sem_color_dict:
      # if I have a dict, make it
      max_sem_key = 0
      for key, data in sem_color_dict.items():
        if key + 1 > max_sem_key:
          max_sem_key = key + 1
      self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
      for key, value in sem_color_dict.items():
        self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
    else:
      # otherwise make random
      max_sem_key = max_classes
      self.sem_color_lut = np.random.uniform(low=0.0,
                                             high=1.0,
                                             size=(max_sem_key, 3))
      # force zero to a gray-ish color
      self.sem_color_lut[0] = np.full((3), 0.1)

    # make instance colors
    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0,
                                            high=1.0,
                                            size=(max_inst_id, 3))
    # force zero to a gray-ish color
    self.inst_color_lut[0] = np.full((3), 0.1)

  def reset(self):
    """ Reset scan members. """
    super(SemLaserScan, self).reset()

    # semantic labels
    self.sem_label = np.zeros((0, 1), dtype=np.int32)          # [m, 1]: label
    self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # instance labels
    self.inst_label = np.zeros((0, 1), dtype=np.int32)          # [m, 1]: label
    self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # projection color with semantic labels
    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                   dtype=np.int32)              # [H,W]  label
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                   dtype=np.float)              # [H,W,3] color

    # projection color with instance labels
    self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                    dtype=np.int32)              # [H,W]  label
    self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                    dtype=np.float)              # [H,W,3] color
  def open_label(self, filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.int32)
    label = label.reshape((-1))

    # set it
    self.set_label(label)

  def set_label(self, label):
    """ Set points for label not from file but from np
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == self.points.shape[0]:
      self.sem_label = label & 0xFFFF  # semantic label in lower half
      self.inst_label = label >> 16    # instance id in upper half
    else:
      print("Points shape: ", self.points.shape)
      print("Label shape: ", label.shape)
      raise ValueError("Scan and Label don't contain same number of points")

    # sanity check
    assert((self.sem_label + (self.inst_label << 16) == label).all())

    if self.project:
      self.do_label_projection()

  def colorize(self):
    """ Colorize pointcloud with the color of each semantic label
    """
    self.sem_label_color = self.sem_color_lut[self.sem_label]
    self.sem_label_color = self.sem_label_color.reshape((-1, 3))
    self.inst_label_color = self.inst_color_lut[self.inst_label]
    self.inst_label_color = self.inst_label_color.reshape((-1, 3))
  def do_label_projection(self):
    # only map colors to labels that exist
    mask = self.proj_idx >= 0

    # semantics
    self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
    self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]
    # instances
    self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
    self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]


EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               max_points=150000,   # max number of points present in dataset
               gt=True):            # send ground truth?
    # save deats
    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.label_files = []

    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))

      print("parsing seq {}".format(seq))

      # get paths for each
      scan_path = os.path.join(self.root, seq, "velodyne")
      label_path = os.path.join(self.root, seq, "labels")

      # get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_label(f)]

      # check all scans have labels
      if self.gt:
        assert(len(scan_files) == len(label_files))

      # extend list
      self.scan_files.extend(scan_files)
      self.label_files.extend(label_files)

    # sort for correspondance
    self.scan_files.sort()
    self.label_files.sort()

    print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                    self.sequences))

  def __getitem__(self, index):
    # get item in tensor shape
    scan_file = self.scan_files[index]
    if self.gt:
      label_file = self.label_files[index]

    # open a semantic laserscan
    if self.gt:
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)

    # open and obtain scan
    scan.open_scan(scan_file)
    if self.gt:
      scan.open_label(label_file)
      # map unused classes to used classes (also for projection)
      scan.sem_label = self.map(scan.sem_label, self.learning_map)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

    # make a tensor of the uncompressed data (with the max num points)
    unproj_n_points = scan.points.shape[0]

    unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)

    # get points and labels
    proj_range = torch.from_numpy(scan.proj_range).clone()
    # print("scan.proj_xyz: ",scan.proj_xyz)
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
    # print("proj_xyz: ", proj_xyz)
    proj_remission = torch.from_numpy(scan.proj_remission).clone()
    proj_mask = torch.from_numpy(scan.proj_mask)
    proj_x = torch.full([self.max_points], -1, dtype=torch.long)
    proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
    proj_y = torch.full([self.max_points], -1, dtype=torch.long)
    proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
    proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()])
    proj = (proj - self.sensor_img_means[:, None, None]
            ) / self.sensor_img_stds[:, None, None]
    proj = proj * proj_mask.float()

    # get name and sequence
    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".bin", ".label")
    # return
    # return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points
    return proj, proj_mask, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, proj_remission, unproj_n_points
  def __len__(self):
    return len(self.scan_files)

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]


class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               sensor,            # sensor to use
               max_points,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True):  # shuffle training set?
    super(Parser, self).__init__()

    # if I am training, get the dataset
    self.root = root
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.max_points = max_points
    self.batch_size = batch_size
    self.workers = workers
    self.gt = gt
    self.shuffle_train = shuffle_train

    # number of classes that matters is the one for xentropy
    self.nclasses = len(self.learning_map_inv)
    
    if self.test_sequences:
      self.test_dataset = SemanticKitti(root=self.root,
                                        sequences=self.test_sequences,
                                        labels=self.labels,
                                        color_map=self.color_map,
                                        learning_map=self.learning_map,
                                        learning_map_inv=self.learning_map_inv,
                                        sensor=self.sensor,
                                        max_points=max_points,
                                        gt=False)

      self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.workers,
                                                    pin_memory=True,
                                                    drop_last=True)
      assert len(self.testloader) >= 0
      self.testiter = iter(self.testloader)

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return SemanticKitti.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.learning_map)

  def to_color(self, label):
    # put label in original values
    label = SemanticKitti.map(label, self.learning_map_inv)
    # put label in color
    return SemanticKitti.map(label, self.color_map)


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
      # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
  x_coord = torch.arange(kernel_size)
  x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
  y_grid = x_grid.t()
  xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

  mean = (kernel_size - 1) / 2.
  variance = sigma**2.

  # Calculate the 2-dimensional gaussian kernel which is
  # the product of two gaussian distributions for two different
  # variables (in this case called x and y)
  gaussian_kernel = (1. / (2. * math.pi * variance)) *\
      torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))

  # Make sure sum of values in gaussian kernel equals 1.
  gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

  # Reshape to 2d depthwise convolutional weight
  gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)

  return gaussian_kernel


class KNN(nn.Module):
  def __init__(self, params, nclasses):
    super().__init__()
    print("*"*80)
    print("Cleaning point-clouds with kNN post-processing")
    self.knn = params["knn"]
    self.search = params["search"]
    self.sigma = params["sigma"]
    self.cutoff = params["cutoff"]
    self.nclasses = nclasses
    print("kNN parameters:")
    print("knn:", self.knn)
    print("search:", self.search)
    print("sigma:", self.sigma)
    print("cutoff:", self.cutoff)
    print("nclasses:", self.nclasses)
    print("*"*80)

  def forward(self, proj_range, unproj_range, proj_argmax, px, py):
    ''' Warning! Only works for un-batched pointclouds.
        If they come batched we need to iterate over the batch dimension or do
        something REALLY smart to handle unaligned number of points in memory
    '''
    # get device
    if proj_range.is_cuda:
      device = torch.device("cuda")
    else:
      device = torch.device("cpu")

    # sizes of projection scan
    H, W = proj_range.shape

    # number of points
    P = unproj_range.shape

    # check if size of kernel is odd and complain
    if (self.search % 2 == 0):
      raise ValueError("Nearest neighbor kernel must be odd number")

    # calculate padding
    pad = int((self.search - 1) / 2)

    # unfold neighborhood to get nearest neighbors for each pixel (range image)
    proj_unfold_k_rang = F.unfold(proj_range[None, None, ...],
                                  kernel_size=(self.search, self.search),
                                  padding=(pad, pad))

    # index with px, py to get ALL the pcld points
    idx_list = py * W + px
    unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list]

    # WARNING, THIS IS A HACK
    # Make non valid (<0) range points extremely big so that there is no screwing
    # up the nn self.search
    unproj_unfold_k_rang[unproj_unfold_k_rang < 0] = float("inf")

    # now the matrix is unfolded TOTALLY, replace the middle points with the actual range points
    center = int(((self.search * self.search) - 1) / 2)
    unproj_unfold_k_rang[:, center, :] = unproj_range

    # now compare range
    k2_distances = torch.abs(unproj_unfold_k_rang - unproj_range)

    # make a kernel to weigh the ranges according to distance in (x,y)
    # I make this 1 - kernel because I want distances that are close in (x,y)
    # to matter more
    inv_gauss_k = (
        1 - get_gaussian_kernel(self.search, self.sigma, 1)).view(1, -1, 1)
    inv_gauss_k = inv_gauss_k.to(device).type(proj_range.type())

    # apply weighing
    k2_distances = k2_distances * inv_gauss_k

    # find nearest neighbors
    _, knn_idx = k2_distances.topk(
        self.knn, dim=1, largest=False, sorted=False)

    # do the same unfolding with the argmax
    proj_unfold_1_argmax = F.unfold(proj_argmax[None, None, ...].float(),
                                    kernel_size=(self.search, self.search),
                                    padding=(pad, pad)).long()
    unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list]

    # get the top k predictions from the knn at each pixel
    knn_argmax = torch.gather(
        input=unproj_unfold_1_argmax, dim=1, index=knn_idx)

    # fake an invalid argmax of classes + 1 for all cutoff items
    if self.cutoff > 0:
      knn_distances = torch.gather(input=k2_distances, dim=1, index=knn_idx)
      knn_invalid_idx = knn_distances > self.cutoff
      knn_argmax[knn_invalid_idx] = self.nclasses

    # now vote
    # argmax onehot has an extra class for objects after cutoff
    knn_argmax_onehot = torch.zeros(
        (1, self.nclasses + 1, P[0]), device=device).type(proj_range.type())
    ones = torch.ones_like(knn_argmax).type(proj_range.type())
    knn_argmax_onehot = knn_argmax_onehot.scatter_add_(1, knn_argmax, ones)

    # now vote (as a sum over the onehot shit)  (don't let it choose unlabeled OR invalid)
    knn_argmax_out = knn_argmax_onehot[:, 1:-1].argmax(dim=1) + 1

    # reshape again
    knn_argmax_out = knn_argmax_out.view(P)

    return knn_argmax_out

class LocallyConnectedXYZLayer(nn.Module):
  def __init__(self, h, w, sigma, nclasses):
    super().__init__()
    # size of window
    self.h = h
    self.padh = h//2
    self.w = w
    self.padw = w//2
    assert(self.h % 2 == 1 and self.w % 2 == 1)  # window must be odd
    self.sigma = sigma
    self.gauss_den = 2 * self.sigma**2
    self.nclasses = nclasses

  def forward(self, xyz, softmax, mask):
    # softmax size
    N, C, H, W = softmax.shape

    # make sofmax zero everywhere input is invalid
    softmax = softmax * mask.unsqueeze(1).float()

    # get x,y,z for distance (shape N,1,H,W)
    x = xyz[:, 0].unsqueeze(1)
    y = xyz[:, 1].unsqueeze(1)
    z = xyz[:, 2].unsqueeze(1)

    # im2col in size of window of input (x,y,z separately)
    window_x = F.unfold(x, kernel_size=(self.h, self.w),
                        padding=(self.padh, self.padw))
    center_x = F.unfold(x, kernel_size=(1, 1),
                        padding=(0, 0))
    window_y = F.unfold(y, kernel_size=(self.h, self.w),
                        padding=(self.padh, self.padw))
    center_y = F.unfold(y, kernel_size=(1, 1),
                        padding=(0, 0))
    window_z = F.unfold(z, kernel_size=(self.h, self.w),
                        padding=(self.padh, self.padw))
    center_z = F.unfold(z, kernel_size=(1, 1),
                        padding=(0, 0))

    # sq distance to center (center distance is zero)
    unravel_dist2 = (window_x - center_x)**2 + \
        (window_y - center_y)**2 + \
        (window_z - center_z)**2

    # weight input distance by gaussian weights
    unravel_gaussian = torch.exp(- unravel_dist2 / self.gauss_den)

    # im2col in size of window of softmax to reweight by gaussian weights from input
    cloned_softmax = softmax.clone()
    for i in range(self.nclasses):
      # get the softmax for this class
      c_softmax = softmax[:, i].unsqueeze(1)
      # unfold this class to weigh it by the proper gaussian weights
      unravel_softmax = F.unfold(c_softmax,
                                 kernel_size=(self.h, self.w),
                                 padding=(self.padh, self.padw))
      unravel_w_softmax = unravel_softmax * unravel_gaussian
      # add dimenssion 1 to obtain the new softmax for this class
      unravel_added_softmax = unravel_w_softmax.sum(dim=1).unsqueeze(1)
      # fold it and put it in new tensor
      added_softmax = unravel_added_softmax.view(N, H, W)
      cloned_softmax[:, i] = added_softmax

    return cloned_softmax


class CRF(nn.Module):
  def __init__(self, params, nclasses):
    super().__init__()
    self.params = params
    self.iter = torch.nn.Parameter(torch.tensor(params["iter"]),
                                   requires_grad=False)
    self.lcn_size = torch.nn.Parameter(torch.tensor([params["lcn_size"]["h"],
                                                     params["lcn_size"]["w"]]),
                                       requires_grad=False)
    self.xyz_coef = torch.nn.Parameter(torch.tensor(params["xyz_coef"]),
                                       requires_grad=False).float()
    self.xyz_sigma = torch.nn.Parameter(torch.tensor(params["xyz_sigma"]),
                                        requires_grad=False).float()

    self.nclasses = nclasses
    print("Using CRF!")

    # define layers here
    # compat init
    self.compat_kernel_init = np.reshape(np.ones((self.nclasses, self.nclasses)) -
                                         np.identity(self.nclasses),
                                         [self.nclasses, self.nclasses, 1, 1])

    # bilateral compatibility matrixes
    self.compat_conv = nn.Conv2d(self.nclasses, self.nclasses, 1)
    self.compat_conv.weight = torch.nn.Parameter(torch.from_numpy(
        self.compat_kernel_init).float() * self.xyz_coef, requires_grad=True)

    # locally connected layer for message passing
    self.local_conn_xyz = LocallyConnectedXYZLayer(params["lcn_size"]["h"],
                                                   params["lcn_size"]["w"],
                                                   params["xyz_coef"],
                                                   self.nclasses)

  def forward(self, input, softmax, mask):
    # use xyz
    xyz = input[:, 1:4]

    # iteratively
    for iter in range(self.iter):
      # message passing as locally connected layer
      locally_connected = self.local_conn_xyz(xyz, softmax, mask)

      # reweigh with the 1x1 convolution
      reweight_softmax = self.compat_conv(locally_connected)

      # add the new values to the original softmax
      reweight_softmax = reweight_softmax + softmax

      # lastly, renormalize
      softmax = F.softmax(reweight_softmax, dim=1)

    return softmax


class Segmentator(nn.Module):
  def __init__(self, ARCH, nclasses, path=None, path_append="", strict=False):
    super().__init__()
    self.ARCH = ARCH
    self.nclasses = nclasses
    self.path = path
    self.path_append = path_append
    self.strict = False

    # get the model
    bboneModule = imp.load_source("bboneModule",
                                  '/home/khanh/catkin_ws/src/lidar_bonnetal/src/train/backbones/darknet.py')
    self.backbone = bboneModule.Backbone(params=self.ARCH["backbone"])

    # do a pass of the backbone to initialize the skip connections
    stub = torch.zeros((1,
                        self.backbone.get_input_depth(),
                        self.ARCH["dataset"]["sensor"]["img_prop"]["height"],
                        self.ARCH["dataset"]["sensor"]["img_prop"]["width"]))

    if torch.cuda.is_available():
      stub = stub.cuda()
      self.backbone.cuda()
    _, stub_skips = self.backbone(stub)

    decoderModule = imp.load_source("decoderModule",
                                    '/home/khanh/catkin_ws/src/lidar_bonnetal/src/train/tasks/semantic/decoders/darknet.py')
    self.decoder = decoderModule.Decoder(params=self.ARCH["decoder"],
                                         stub_skips=stub_skips,
                                         OS=self.ARCH["backbone"]["OS"],
                                         feature_depth=self.backbone.get_last_depth())

    self.head = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                              nn.Conv2d(self.decoder.get_last_depth(),
                                        self.nclasses, kernel_size=3,
                                        stride=1, padding=1))

    if self.ARCH["post"]["CRF"]["use"]:
      self.CRF = CRF(self.ARCH["post"]["CRF"]["params"], self.nclasses)
    else:
      self.CRF = None

    # print number of parameters and the ones requiring gradients
    # print number of parameters and the ones requiring gradients
    weights_total = sum(p.numel() for p in self.parameters())
    weights_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)

    # breakdown by layer
    weights_enc = sum(p.numel() for p in self.backbone.parameters())
    weights_dec = sum(p.numel() for p in self.decoder.parameters())
    weights_head = sum(p.numel() for p in self.head.parameters())
    print("Param encoder ", weights_enc)
    print("Param decoder ", weights_dec)
    print("Param head ", weights_head)
    if self.CRF:
      weights_crf = sum(p.numel() for p in self.CRF.parameters())
      print("Param CRF ", weights_crf)

    # get weights
    if path is not None:
      # try backbone
      try:
        w_dict = torch.load(path + "/backbone" + path_append,
                            map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model backbone weights")
      except Exception as e:
        print()
        print("Couldn't load backbone, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try decoder
      try:
        w_dict = torch.load(path + "/segmentation_decoder" + path_append,
                            map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model decoder weights")
      except Exception as e:
        print("Couldn't load decoder, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try head
      try:
        w_dict = torch.load(path + "/segmentation_head" + path_append,
                            map_location=lambda storage, loc: storage)
        self.head.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try CRF
      if self.CRF:
        try:
          w_dict = torch.load(path + "/segmentation_CRF" + path_append,
                              map_location=lambda storage, loc: storage)
          self.CRF.load_state_dict(w_dict, strict=True)
          print("Successfully loaded model CRF weights")
        except Exception as e:
          print("Couldn't load CRF, using random weights. Error: ", e)
          if strict:
            print("I'm in strict mode and failure to load weights blows me up :)")
            raise e
    else:
      print("No path to pretrained, using random init.")

  def forward(self, x, mask=None):
    y, skips = self.backbone(x)
    y = self.decoder(y, skips)
    y = self.head(y)
    y = F.softmax(y, dim=1)
    if self.CRF:
      assert(mask is not None)
      y = self.CRF(x, y, mask)
    return y

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir

    # get the data
    self.parser = Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = Segmentator(self.ARCH,
                               self.parser.get_n_classes(),
                               self.modeldir)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()
  def calculating_bin(self, loader, to_orig_fn):
    for i, (proj_in, proj_mask, path_seq, path_name, p_x, p_y, proj_range, unproj_range, proj_xyz, proj_remission, npoints) in enumerate(loader):
      # first cut to rela size (batch size one allows it)
      proj_remission = proj_remission[0, :npoints]
      proj_xyz = proj_xyz[0, :npoints]
      p_x = p_x[0, :npoints]
      p_y = p_y[0, :npoints]
      proj_range = proj_range[0, :npoints]
      unproj_range = unproj_range[0, :npoints]
      path_seq = path_seq[0]
      path_name = path_name[0]
      if self.gpu:
        proj_in = proj_in.cuda()
        proj_mask = proj_mask.cuda()
        p_x = p_x.cuda()
        p_y = p_y.cuda()  
        if self.post:
          proj_range = proj_range.cuda()
          unproj_range = unproj_range.cuda()
      # compute output
      proj_output = self.model(proj_in, proj_mask)
      # print (proj_output.shape)
      proj_argmax = proj_output[0].argmax(dim=0)
            
      # rospy.init_node('talker', anonymous=True) 
      # msg = dulieu()
      list = []  

      for r in range(16):          
        for i, index in zip(proj_argmax[r], range(len(proj_argmax[r]))):    
          if i == 15:
            msg.x = proj_xyz[r][index][0].item()
            msg.y = proj_xyz[r][index][1].item()
            msg.z = proj_xyz[r][index][2].item()
            msg.r = proj_range[r][index].item()
            msg.i = proj_remission[r][index].item()
            pub.publish(msg)
            list.append(proj_xyz[r][index][0].item())
      # print(len(list))
      msg.n = len(list)
      # print(msg.n)
      list.clear()       

      print("--------------------------------")
      # print("proj_argmax: ", proj_argmax.shape) #Output là màu của các điểm khi áp dụng Spherical Projection

      if self.post:
        # knn postproc
        unproj_argmax = self.post(proj_range,
                                        unproj_range,
                                        proj_argmax,
                                        p_x,
                                        p_y)
      else:
        # put in original pointcloud using indexes
        unproj_argmax = proj_argmax[p_y, p_x]
        # print("unproj_argmax_2: ", unproj_argmax[p_x].shape)
      # print(unproj_argmax[0], unproj_argmax[1], unproj_argmax[2])
      # measure elapsed time
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      end = time.time()
      print("Infered seq", path_seq, "scan", path_name,
                  "in", time.time() - end, "sec")
      

      # save scan
      # get the first scan in batch and project scan
      pred_np = unproj_argmax.cpu().numpy()
      pred_np = pred_np.reshape((-1)).astype(np.int32)
      # map to original label
      pred_np = to_orig_fn(pred_np)
      # print (pred_np.shape)

      # save scan
      path = os.path.join(self.logdir, "sequences",
                                path_seq, "predictions", path_name)
      pred_np.tofile(path)   

  def infer_subset(self):
    # switch to evaluate mode
    temp_int = 0
    while True:
      if os.path.exists('/home/khanh/catkin_ws/src/lidar_bonnetal/src/train/tasks/semantic/data/sequences/23/velodyne_1/000000.bin'):
        while True:
          # time.sleep(10)
          loader=self.parser.get_test_set()
          to_orig_fn=self.parser.to_original
          temp_str = str(temp_int)
          temp_str_zero = temp_str.zfill(6)
          temp_str_multi = '/home/khanh/catkin_ws/src/lidar_bonnetal/src/train/tasks/semantic/data/sequences/23/velodyne_1/' + temp_str_zero + '.bin'
          temp_str_multi_copy = '/home/khanh/catkin_ws/src/lidar_bonnetal/src/train/tasks/semantic/data/sequences/23/velodyne/' + temp_str_zero + '.bin'
          temp_str_multi_copy_rename = '/home/khanh/catkin_ws/src/lidar_bonnetal/src/train/tasks/semantic/data/sequences/23/velodyne/000000.bin'
          shutil.copyfile(temp_str_multi, temp_str_multi_copy)
          os.rename(temp_str_multi_copy,temp_str_multi_copy_rename)
          self.parser.test_dataset = SemanticKitti(root=self.parser.root,
                                            sequences=self.parser.test_sequences,
                                            labels=self.parser.labels,
                                            color_map=self.parser.color_map,
                                            learning_map=self.parser.learning_map,
                                            learning_map_inv=self.parser.learning_map_inv,
                                            sensor=self.parser.sensor,
                                            max_points=self.parser.max_points,
                                            gt=False)

          self.parser.testloader = torch.utils.data.DataLoader(self.parser.test_dataset,
                                                        batch_size=self.parser.batch_size,
                                                        shuffle=False,
                                                        num_workers=self.parser.workers,
                                                        pin_memory=True,
                                                        drop_last=True)

          print(len(self.parser.testloader))
          
          # loader=self.parser.get_test_set()
          # to_orig_fn=self.parser.to_original
          # temp_str = str(temp_int)
          # temp_str_zero = temp_str.zfill(6)
          # temp_str_multi = '/home/khanh/catkin_ws/src/lidar_bonnetal/src/train/tasks/semantic/data/sequences/23/velodyne/' + temp_str_zero + '.bin'
          # if os.path.exists('/home/khanh/catkin_ws/src/lidar_bonnetal/src/train/tasks/semantic/data/sequences/23/velodyne/000000.bin'):
          if os.path.exists(temp_str_multi_copy_rename):
            self.model.eval()
            if self.gpu:
              torch.cuda.empty_cache()
            
            with torch.no_grad():
              end = time.time()
            self.calculating_bin(loader,to_orig_fn)             
            # os.remove(temp_str_multi_copy_rename)
          else:
            print("Kiem tra 2")
          temp_int = temp_int + 1
          temp_str = str(temp_int)
          temp_str_zero = temp_str.zfill(6)
          temp_str_multi = '/home/khanh/catkin_ws/src/lidar_bonnetal/src/train/tasks/semantic/data/sequences/23/velodyne_1/' + temp_str_zero + '.bin'
          print(temp_str_multi)
          while True:
            if os.path.exists(temp_str_multi):
              os.remove(temp_str_multi_copy_rename)    
              break
            else:
              time.sleep(1)      
      else:
        continue
if __name__ == '__main__':
  
  pub = rospy.Publisher('chatter', dulieu, queue_size=10)     
  rospy.init_node('talker', anonymous=True) 
  msg = dulieu()
  parser = argparse.ArgumentParser("./infer.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to train with. No Default',
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default=os.path.expanduser("~") + '/logs/' +
      datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
      help='Directory to put the predictions. Default: ~/logs/date+time'
  )
  parser.add_argument(
      '--model', '-m',
      type=str,
      required=True,
      default=None,
      help='Directory to get the trained model.'
  )
  FLAGS, unparsed = parser.parse_known_args()

  # open arch config file
  try:
    print("Opening arch config file from %s" % FLAGS.model)
    ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  # open data config file
  try:
    print("Opening data config file from %s" % FLAGS.model)
    DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()

  # create log folder
  try:
    if os.path.isdir(FLAGS.log):
      shutil.rmtree(FLAGS.log)
    os.makedirs(FLAGS.log)
    os.makedirs(os.path.join(FLAGS.log, "sequences"))

    for seq in DATA["split"]["test"]:
      seq = '{0:02d}'.format(int(seq))
      print("test", seq)
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    raise

  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    quit()

  # does model folder exist?
  if os.path.isdir(FLAGS.model):
    print("model folder exists! Using model from %s" % (FLAGS.model))
  else:
    print("model folder doesnt exist! Can't infer...")
    quit()

  # create user and infer dataset
  user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model)
  # user.infer()
  user.infer_subset()

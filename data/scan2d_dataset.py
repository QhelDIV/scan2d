"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
import numpy as np
import h5py
# from data.image_folder import make_dataset
# from PIL import Image
def normalize(x):
    return x/np.sqrt((x*x).sum())
def length(x):
    return np.sqrt((x*x).sum())

class Scan2dDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        ## get the image paths of your dataset;
        #self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        ## define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        #self.transform = get_transform(opt)
        
        with h5py.File('digit6_211.h5','r') as f:
            self.camera_poses = f['camera_pos'][0]
            self.camera_dirs = f['camera_dir'][0]
            self.depths = f['depth'][0]
        self.length = self.camera_poses.shape[0]
        self.perm   = np.random.permutation(self.length)
                

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        return {'camera_pos':   self.camera_poses[self.perm[index]], \
                'depth':        self.depths[self.perm[index]], \
                'original_index': self.perm[index]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.length)

def genRays(num_rays=64,position = np.array([99,0]), lookat_dir=None, fov=np.pi/2.):
    if lookat_dir is None:
        print("ERROR! no lookat_dir is given!")
    ray_dirs=[]
    phi = fov
    depths = np.zeros(num_rays)
    angle_step = phi / (num_rays + 1)
    angle = phi/2. - angle_step/2.
    for i in range(num_rays):
        cos, sin = np.cos(angle), np.sin(angle)
        R = np.array([[cos,-sin],[sin,cos]])
        ray_dir = np.dot(R, lookat_dir)
        ray_dirs.append(ray_dir)
        #print(world_look_dir)
        #print(look_at - position)
        #print(R)
        #print(ray_dir, cos, sin, angle)
        angle -= angle_step
    return np.array(ray_dirs)

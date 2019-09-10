import numpy as np
import h5py
from PIL import Image
from config import config

infinity = 100000000.

def save_images(images):
    for i in range(images.shape[0]):
        image = Image.fromarray(images[i]*255).convert("L")
        image.save('images/digit6/%d.png'%i)
def normalize(x):
    return x/np.sqrt((x*x).sum())
def length(x):
    return np.sqrt((x*x).sum())
def generate_cams_matrix(image, views=10):
    dim = image.shape[0]
    look_at = np.array([dim/2., dim/2.])

    # position = np.random.rand(2)
    position = np.array([0.,0.])
    world_look_dir = normalize(look_at - position)
    world_up_dir = np.array([-world_look_dir[1], world_look_dir[0]])
    #world_frame_look_dir = world_frame_look_dir/np.sum((world_frame_look_dir**2).sum())

    #cos = cam_frame_look_dir * world_frame_look_dir
    #sin = cam_frame_look_dir[0]*world_frame_look_dir[1] - world_frame_look_dir[0] * cam_frame_look_dir[1]
    #print("cos:%f sin:%f"%(cos,sin))

    # extrinsic
    #R = np.array([[cos,-sin],[sin,cos]])
    #t = position
    # intrinsic
    # K = np.array([  [a_x,   0 ], 
    #                 [0.,    1.]])

    # R_cam     = np.array([  [  world_look_dir[0],    world_look_dir[1],          0  ],
    #                         [    world_up_dir[0],      world_up_dir[1],          0  ],
    #                         [                  0,                  0,            1  ]])
    # t_cam     = np.array([  [                  1,                  0,  -position[0] ],
    #                         [                  0,                  1,  -position[1] ],
    #                         [                  0,                  0,            1  ]])
    # M_cam     = np.dot(R_cam, t_cam)
    # M_cam_inv = np.array([  [  world_look_dir[0],    world_up_dir[0],   position[0] ],
    #                         [  world_look_dir[1],    world_up_dir[1],   position[1] ],
    #                         [                  0,                  0,             1 ]])


    # M_vp = np.array([   [n_x/2.,             0,        (n_x-1.)/2.],
    #                     [     0,             1,                  0],
    #                     [     0,             0,                  1]])
def genRays(num_rays=64,position = np.array([99,0]), lookat_dir=None, fov=np.pi/2.):
    if lookat_dir is None:
        print("ERROR! no lookat_dir is given!")
    position = np.array([0,0])
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
        print(ray_dir)
    exit()
    return np.array(ray_dirs)
def generate_depthmap(image, num_rays=64,position = np.array([99,0]), lookat_dir=None, fov=np.pi/2.):
    imgdim = image.shape[0]
    if lookat_dir is None:
        look_at = np.array([imgdim/2., imgdim/2.])
        lookat_dir = normalize(look_at - position)
        
    depths = np.zeros(num_rays)
    ray_dirs = genRays(num_rays = num_rays, position = position, lookat_dir = lookat_dir, fov=fov)
    for i in range(num_rays):
        raypos = np.copy(position)
        ray_dir = ray_dirs[i]
        print(raypos)
        depth = 0.
        hit = False
        while raypos[0]>=0. and raypos[0]<=imgdim and raypos[1]>=0. and raypos[1]<=imgdim:
            pixel_coordinate = np.floor(raypos).astype(np.int)
            pixel = image[pixel_coordinate[0], pixel_coordinate[1]]
            if pixel > 0.01:
                hit = True
                break
            #print(raypos)
            raypos += ray_dir
            depth += 1
        if hit == True:
            depths[i] = depth
            #print(i, raypos)
        else:
            depths[i] = infinity
    return depths
def generate_cams(image, views):
    imgdim = image.shape[0]
    positions = np.zeros((views, 2))
    cam_directions = np.zeros((views, 2))
    depths    = np.zeros((imgdim, config.num_rays))


    count = 0
    while count<views:
        look_at = np.array([imgdim/2., imgdim/2.])
        #world_look_dir = normalize(look_at - position)
        position = np.random.rand(2) * imgdim
        if length(position-look_at)>.6:
            look_at = np.array([imgdim/2., imgdim/2.])
            lookat_dir = normalize(look_at - position)
            positions[count] = position
            cam_directions[count] = lookat_dir
            depth = generate_depthmap(image, num_rays = config.num_rays, position = position, lookat_dir=lookat_dir)
            if depth.min() < 5.:
                continue
            depths[count] = depth
            count += 1
    for i in range(views):
        cam   = positions[i]
        depth = depths[i]
        hits = np.where(depth<10000.)[0]
        mean = np.mean(depth[hits])
        std  = np.std(depth[hits])
        print("cam %d:%s mindepth:%s hits:%d mean:%f std:%f"%(i, str(cam), str(np.min(depth)), hits.shape[0], mean, std ))
    return positions, cam_directions, depths

def create_single_digit6_dataset():
    with h5py.File('digit6.h5','r+') as f:
        images=np.array(f['images'])
        fonts=np.array(f['fonts'])
    cam_positions = []
    cam_directions = []
    depths = []
    for i in range(1):
        positioni, cam_diri, depthi = generate_cams(images[211], views=100)
        cam_positions.append( positioni )
        cam_directions.append( cam_diri )
        print(positioni[0], cam_diri[0])
        depths.append( depthi )
    print(cam_positions[0][-1], list(depths[0][-1].astype(int)))
    with h5py.File('digit6_211.h5','w') as f:
        f['images']     = images[211:212]
        f['fonts']      = fonts[211:212]
        f['camera_pos'] = np.array(cam_positions)
        f['camera_dir'] = np.array(cam_directions)
        f['depths']     = np.array(depths)

def create_all_digit6_dataset():
    with h5py.File('digit6.h5','r+') as f:
        images=np.array(f['images'])
        fonts=np.array(f['fonts'])
        cam_positions = []
        depths = []
        for i in range(images.shape[0]):
            positioni, directioni, depthi = generate_cams(images[0], views=10)
            cam_positions.append( positioni )
            depths.append( depthi )
        f['camera_pos'] = np.array(cam_positions)
        f['depths']     = np.array(depths)

if __name__=='__main__':
    #create_all_digit6_dataset()
    create_single_digit6_dataset()

        

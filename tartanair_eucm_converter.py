'''
Converting the equirect images from tartanair to more camera models using omnicv.

To do this, we
1. Load an equirect image.
2. Convert it to a cubemap.
3. Convert the cubemap to the desired cubemap format that omnicv expects.
4. Convert the cubemap to the desired camera model using omnicv.
'''

# Load a cubemap.
import os
import sys
sys.path.append('/home/yoraish/code/tartanairpy2')
import tartanair as ta
import cv2
import numpy as np
import sys
from omnicv import fisheyeImgConv

# Load six images.

####################################################################
# PARAMETER SETTING.
####################################################################
####################################################################
env = "AbandonedCableExposure"
difficulty = ['hard']
trajectory_id = ["P002"]
modality = ['image']
camera_name = ['lcam_front', 'lcam_left', 'lcam_right', 'lcam_back', 'lcam_top', 'lcam_bottom']
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2'

####################################################################
####################################################################
output_dir_root = f'/home/yoraish/code/tartanvo-fisheye-transformer/src/tartanvo_fisheye/evaluation/baselines/fisheye-ORB-SLAM/data/tartanair_eucm/{env}/Data_{difficulty[0]}/{trajectory_id[0]}/'
output_dir_data = os.path.join(output_dir_root, 'data')
output_txt_fpath = os.path.join(output_dir_root, 'rgb.txt')

ta.init(tartanair_data_root)
dataloader = ta.create_image_slowloader(env, difficulty, trajectory_id, modality, camera_name, batch_size=1, shuffle=False, num_workers=1)

# Fisheye conversion parameters..
# H_fish, W_fish = 240, 240
H_fish, W_fish = 240, 240
equirect_shape = (1000, 2000)

# Params for EUNC.
f_eunc = 120
alpha_eunc = 0.99
beta_eunc = 1.0
angles_eunc = [0, 0, -45]
                
# Params for DS.                                                                                
f_DS = 250
alpha_DS = 0.6
xi_DS = -0.2
angles_DS = [0, 0, -45]
mapper = fisheyeImgConv()



def batch_sample_to_cube(batch, b):
    img_front = batch['rgb_lcam_front'][b][0]
    img_right = batch['rgb_lcam_right'][b][0]
    img_left = batch['rgb_lcam_left'][b][0]
    img_back = batch['rgb_lcam_back'][b][0]
    img_top = batch['rgb_lcam_top'][b][0]
    img_bottom = batch['rgb_lcam_bottom'][b][0]

    # Convert to an image cube.
    # The shape of the image cube is (3H, 4W, C).
    H, W, C = img_front.shape
    img_cube = np.zeros((3*H, 4*W, C), dtype = np.uint8)
    img_cube[H:2*H, 0:W, :] = img_left
    img_cube[H:2*H, W:2*W, :] = img_front
    img_cube[H:2*H, 2*W:3*W, :] = img_right
    img_cube[H:2*H, 3*W : 4*W, :] = img_back
    img_cube[0:H, W : 2*W, :] = img_top
    img_cube[2*H:3*H, W : 2*W, :] = img_bottom

    # Do this in a way that conforms with the omnicv cubemap format.
    # The format is:
    # |      |  top  |       |      |
    # | back |  left | front | right| 
    # |      | bottom|       |      |

    img_cube = np.zeros((3*H, 4*W, C), dtype = np.uint8)
    img_cube[H:2*H, 0:W, :] = img_back
    img_cube[H:2*H, W:2*W, :] = img_left
    img_cube[H:2*H, 2*W:3*W, :] = img_front
    img_cube[H:2*H, 3*W : 4*W, :] = img_right
    img_cube[0:H, W : 2*W, :] = np.rot90(img_top, 1)
    img_cube[2*H:3*H, W : 2*W, :] = np.rot90(img_bottom, 3)

    return img_cube

# Open the text file for writing.
f = open(output_txt_fpath, 'w')

sample_ix = 0 # The format of the saved stamp is 1520531829351146058 (19 digits for the seconds, first 9 digits for the seconds, last 10 digits for the nanoseconds). 

# Iterate over the batches.
for i in range(500):    
    # Get the next batch.
    batch = dataloader.load_sample()
    # Check if the batch is None.
    if batch is None:
        break
    batch_size = batch['rgb_lcam_front'].shape[0]
    # Visualize some images.
    # The shape of an image batch is (B, S, H, W, C), where B is the batch size, S is the sequence length, H is the height, W is the width, and C is the number of channels.
    for b in range(batch_size):
        img_cube = batch_sample_to_cube(batch, b)

        # Convert to equirect.
        equirect = mapper.cubemap2equirect(img_cube, equirect_shape)

        # Convert to a fisheye image.
        # fisheye = mapper.equirect2Fisheye_EUCM(equirect, outShape=[H_fish, W_fish], f=f_eunc, a_=alpha_eunc, b_=beta_eunc, angles=angles_eunc)

        # fisheye = mapper.equirect2Fisheye_DS(equirect, outShape=[H_fish, W_fish], f=f_DS, a_=alpha_DS, xi_=xi_DS, angles= angles_DS)

        fisheye = mapper.equirect2Fisheye_EUCM(equirect, outShape=[H_fish, W_fish], f=f_eunc, a_=alpha_eunc, b_=beta_eunc, angles=angles_eunc)
        
        # Mirror.
        fisheye = cv2.flip(fisheye, 1)
        fisheye = cv2.cvtColor(fisheye, cv2.COLOR_RGB2BGR)


        # Save the fisheye image. The name of the image is the stamp.
        stamp_str = "{:019d}".format(sample_ix * 100000000)
        sample_ix += 1
        img_fpath = os.path.join(output_dir_data, stamp_str + '.png')
        cv2.imwrite(img_fpath, fisheye)

        # Write to the text file.
        # The format of the text file is:
        # timestamp filename
        f.write(stamp_str + ' data/' + stamp_str + '.png\n')

        winname = "image: f = {}, alpha = {}, beta = {}".format(f_eunc, alpha_eunc, beta_eunc)
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.imshow(winname, fisheye)
        cv2.waitKey(1)

'''
A quick script to get the ATE and RPE scores from a given orb-slam generated sequence.
The sequence is of the form
frame-index x y z qx qy qz qw

The way we'll carry out the evaluation would be by first loading both the ORB estimated trajectory and also the ground truth trajectory.
Then, we'll only keep the poses from the relevant frames in the ground truth trajectory.
Finally, we'll compute the ATE and RPE scores.
'''

import os
from colorama import Fore, Style
import numpy as np
from scipy.spatial.transform import Rotation
import sys
sys.path.append('/home/yoraish/code/tartanairpy2')
import tartanair as ta
ta.init('/media/yoraish/overflow/data/tartanair-v2')


####################################################################
# PARAMETER SETTING.
####################################################################

# Get the ground truth trajectory.
env = "ShoreCavesExposure"
difficulty = "hard"
seq = "P001"
camera_name = "lcam_front"
use_only_keyframes = False

data_dir = f'/home/yoraish/code/tartanvo-fisheye-transformer/src/tartanvo_fisheye/evaluation/baselines/fisheye-ORB-SLAM/data/tartanair_eucm/{env}/Data_{difficulty}/{seq}/results/'
orb_traj_path = os.path.join(data_dir, 'est_orb.txt')
tvofe_traj_path = os.path.join(data_dir, 'est_tvofe.txt')

plot_out_path_orb = os.path.join(data_dir, 'plot_orb.png')
plot_out_path_tvofe = os.path.join(data_dir, 'plot_tvofe.png')

####################################################################
####################################################################


####################
# Get the trajectories
####################
# Get the ground truth trajectory.
gt_traj = ta.get_traj_np(env, difficulty, seq, camera_name)

# Get the ORB trajectory.
orb_traj = np.loadtxt(orb_traj_path)


# Filter the ground truth trajectory to only include the relevant frames.
orb_frame_ixs = orb_traj[:, 0].astype(int)
gt_traj_filt = gt_traj[orb_frame_ixs, :]

# Only keep the relevant columns of the ORB trajectory. Meaning that we take out the first one.
orb_first_frame = int(orb_traj[0, 0])
orb_last_frame =  int(orb_traj[-1, 0])
orb_traj = orb_traj[:, 1:]

# We must rotate all the poses estimated by Fisheye-ORB-SLAM from RDF (Z front, X right, Y down) to NED (X front, Y right, Z down).
R_ned_rdf = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])


# R_ned_rdf = Rotation.from_euler('xyz', [0, -90, 0], degrees=True).as_matrix()

# for i in range(orb_traj.shape[0]):
#     R_i = Rotation.from_quat(orb_traj[i, 3:]).as_matrix()
#     R_i = R_ned_rdf @ R_i
#     q_i = Rotation.from_matrix(R_i).as_quat()
#     orb_traj[i, 3:] = q_i

# Also rotate the entire estimated trajectory.
orb_traj[:, :3] = (R_ned_rdf @ orb_traj[:, :3].T).T
####################
# Compute the ATE and RPE scores.
####################
result_orb = ta.evaluate_traj(orb_traj, gt_traj_filt, plot=False, plot_out_path = plot_out_path_orb)

####################
# Optionally, look at the TARTANVO_FISHEYE estimated trajectory and evaluate it in the subset available in the ORB-SLAM tracked-section.
# This would make the case "TVO is better not only because it maintains tracking, but also because it's more accurate in the tracked sections of ORB."
####################

# Get the TARTANVO_FISHEYE trajectory.
tvofe_traj = np.loadtxt(tvofe_traj_path)

if not use_only_keyframes:
    # Filter the ground truth trajectory to only include the relevant frames.
    gt_traj_filt = gt_traj[orb_first_frame: orb_last_frame - 5, :]
    tvofe_traj_filt = tvofe_traj[orb_first_frame: orb_last_frame - 5, :]

else:
    # Can also use only the keyframes from ORB.
    # print("USING ONLY KEYFRAMES FROM ORB")
    gt_traj_filt = gt_traj[orb_frame_ixs[:-3]]
    tvofe_traj_filt = tvofe_traj[orb_frame_ixs[:-3]]

    print(Fore.RED + "WARNING: USING ONLY KEYFRAMES FROM ORB" + Style.RESET_ALL)
    print(Fore.RED + "WARNING: USING ONLY KEYFRAMES FROM ORB" + Style.RESET_ALL)
    print(Fore.RED + "WARNING: USING ONLY KEYFRAMES FROM ORB" + Style.RESET_ALL)

# Scale all the orientations.
# tvofe_traj_filt[:, :3] = tvofe_traj_filt[:, :3] * 10

# Compute the ATE and RPE scores.
result_tvofe = ta.evaluate_traj(tvofe_traj_filt, gt_traj_filt, plot=False, plot_out_path = plot_out_path_tvofe)

####################
# Save the results.
####################

# Save the results.
results_path = os.path.join(data_dir, 'results.txt')
with open(results_path, 'w') as f:
    f.write(f"ORB ATE/RPE t/R   : {result_orb['ate']}, {result_orb['rpe']}\n")
    f.write(f"TVOFE ATE/RPE t/R : {result_tvofe['ate']}, {result_tvofe['rpe']}\n")
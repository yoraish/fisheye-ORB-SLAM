'''
A quick script to get the ATE and RPE scores from a given orb-slam generated sequence.
The sequence is of the form
frame-index x y z qx qy qz qw

The way we'll carry out the evaluation would be by first loading both the ORB estimated trajectory and also the ground truth trajectory.
Then, we'll only keep the poses from the relevant frames in the ground truth trajectory.
Finally, we'll compute the ATE and RPE scores.
'''

import numpy as np
import sys
sys.path.append('/home/yoraish/code/tartanairpy2')
import tartanair as ta

ta.init('/media/yoraish/overflow/data/tartanair-v2')

####################
# Get the trajectories
####################

# Get the ground truth trajectory.
env = "MiddleEastExposure"
difficulty = "easy"
seq = "P000"
camera_name = "lcam_front"
gt_traj = ta.get_traj_np(env, difficulty, seq, camera_name)

# Get the ORB trajectory.
orb_traj = np.loadtxt('/home/yoraish/code/tartanvo-fisheye-transformer/src/tartanvo_fisheye/evaluation/baselines/fisheye-ORB-SLAM/saved_keyframe_trajectories/MiddleEastExposure_easy_P000.txt')

# Filter the ground truth trajectory to only include the relevant frames.
gt_traj = gt_traj[orb_traj[:, 0].astype(int), :]

# Only keep the relevant columns of the ORB trajectory. Meaning that we take out the first one.
orb_traj = orb_traj[:, 1:]

####################
# Compute the ATE and RPE scores.
####################

ta.evaluate_traj(orb_traj, gt_traj, plot=False, plot_out_path = 'plot.png')
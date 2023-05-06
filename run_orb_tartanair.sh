ENV_NAME="AbandonedCableExposure"
DIFFICULTY="hard"
TRAJ_NAME="P002"
CONFIG_PATH="data/tartanair_eucm/calib_config/fisheye_eucm_cam0.yaml"
TRAJ_DATA_PATH="data/tartanair_eucm/${ENV_NAME}/Data_${DIFFICULTY}/${TRAJ_NAME}"


# Run.
./Examples/Monocular/mono_tum \Vocabulary/ORBvoc.bin ${CONFIG_PATH} ${TRAJ_DATA_PATH}

# /home/yoraish/code/tartanvo-fisheye-transformer/src/tartanvo_fisheye/evaluation/baselines/fisheye-ORB-SLAM/data/tartanair_eucm/MiddleEastExposure/Data_hard/P000/data/.

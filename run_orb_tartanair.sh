ENV_NAME="MocapEnv"
DIFFICULTY="hard"
TRAJ_NAME="P003"

# Optionally pass as arguments.
# Pass -d for difficulty.
# Pass -t for trajectory name.

# Parse arguments.
while getopts d:t: flag
do
    case "${flag}" in
        d) DIFFICULTY=${OPTARG};;
        t) TRAJ_NAME=${OPTARG};;
    esac
done

echo "Running ORB-SLAM on ${TRAJ_NAME} in ${DIFFICULTY} difficulty."
TRAJ_DATA_PATH="/media/yoraish/overflow/data/fish_mocap/tartanair_converted/${ENV_NAME}/Data_${DIFFICULTY}/${TRAJ_NAME}"
CONFIG_PATH="data/tartanair_eucm/calib_config/fisheye_eucm_cam0.yaml"



# Run.
./Examples/Monocular/mono_tum \Vocabulary/ORBvoc.bin ${CONFIG_PATH} ${TRAJ_DATA_PATH}

# Example: /home/yoraish/code/tartanvo-fisheye-transformer/src/tartanvo_fisheye/evaluation/baselines/fisheye-ORB-SLAM/data/tartanair_eucm/MiddleEastExposure/Data_hard/P000/data/.


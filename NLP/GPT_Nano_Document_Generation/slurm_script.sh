#!/bin/bash
#SBATCH -J CharGPT            # please replace by short unique self speaking identifier UR101ATTN SSONEFORMER MASK2FORMER TestEval
#SBATCH -N 1                  # number of nodes, we only have one
#SBATCH --gres=gpu:v100:7     # type:number of GPU cards 6
#SBATCH --mem-per-cpu=5500    # main MB per task? max. 500GB/80=6GB
#SBATCH --ntasks-per-node 7   # bigger for mpi-tasks 6
#SBATCH --cpus-per-task 10    # 10 CPU-threads needed (physcores*2)
#SBATCH --time 24:00:00       # set days-hrs:min:ss walltime #2-17:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/nfs1/malla/Computer Vision/GPT_Nano_Document_Generation/slurmlogs/%j.out    # STDOUT
#SBATCH --mail-user=bhavana.malla@ovgu.de

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export TMPDIR="/scratch2/malla/tmp/"
echo temp dir: $TMPDIR
echo slurm temp dir: $SLURM_TMPDIR

exec 2>&1      # redirects the standard error (file descriptor 2) to the same location as the standard output (file descriptor 1)
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID

# Program root dir
programROOT=/nfs1/malla/Thesis_copy 
# Script to execute
pythonMain=train_net.py 

#Create full path of the program
pyFullPath=$programROOT/$pythonMain

#Activate conda environment
source /nfs1/malla/anaconda3/etc/profile.d/conda.sh # conda installation
conda activate thesis # environment to be activated

export NCCL_BLOCKING_WAIT=1  # Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname)  # Store the master node’s IP address in the MASTER_ADDR environment variable.
export NCCL_DEBUG=WARN # INFO
export NCCL_P2P_LEVEL=NVL
echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"
echo "r$SLURM_JOB_NUM_NODES Slurm nodes"


# Execute python code
# torchrun --standalone --nproc_per_node=$SLURM_NTASKS $pyFullPath --is_train True
python $pyFullPath --is_train True --encoder "mit_b3" --decoder "FPN" --enable_mcdropout True --enable_ohem True --init_method tcp://$MASTER_ADDR:27614 --world_size $SLURM_NTASKS

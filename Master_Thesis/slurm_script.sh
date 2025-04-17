#!/bin/bash

#SBATCH -J OHEM_Test      # please replace by short unique self speaking identifier UR101ATTN SSONEFORMER MASK2FORMER TestEval
#SBATCH -N 1                  # number of nodes, we only have one
##SBATCH --exclusive
#SBATCH --gres=gpu:v100:1     # type:number of GPU cards 6
#SBATCH --mem-per-cpu=36000    # main MB per task? max. 500GB/80=6GB    5500 in MBS  # 6250 # 32000 for eval single card
#SBATCH --ntasks-per-node 1   # bigger for mpi-tasks 6
#SBATCH --cpus-per-task 5    # 10 CPU-threads needed (physcores*2)  # 10 when 6250 or 5500 # 5 when 32000
#SBATCH --time 1:30:00       # set 0h59min walltime # 48:00:00 for test # 2:30:00 for eval 
#SBATCH --mail-type=END,FAIL
####SBATCH --mail-user=xxx.yyy@zzz.com
#SBATCH --output=/nfs1/malla/Thesis/slurmlogs/test_eval_metrics/%j.out    # valloss_trainings, valmiou_trainings, test_runs, test_eval_metrics, pretrainings, finetune_trainings, predictions

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export TMPDIR="/scratch2/malla/tmp/"
echo temp dir: $TMPDIR
echo slurm temp dir: $SLURM_TMPDIR

# redirects the standard error (file descriptor 2) to the same location as the standard output (file descriptor 1)
exec 2>&1
echo "DEBUG: host=$(hostname) pwd=$(pwd) ulimit=$(ulimit -v) \$1=$1 \$2=$2"
scontrol show Job $SLURM_JOBID          # show slurm-command and more for DBG

#Activate conda environment
source /nfs1/malla/anaconda3/etc/profile.d/conda.sh # conda installation
conda activate thesis                               # environment to be activated

# Use the first available GPU (could also modify to use multiple GPUs if needed)
# export CUDA_VISIBLE_DEVICES=2   # 1,2,3,4
# echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

export NCCL_BLOCKING_WAIT=1                         # Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname)                      # Store the master node’s IP address in the MASTER_ADDR environment variable.
export NCCL_DEBUG=WARN                              # INFO
export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=


echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"
echo "r$SLURM_JOB_NUM_NODES Slurm nodes"

programROOT=/nfs1/malla/Thesis          # program root dir
pythonMain=run.py                       # script to execute

# Create full path of the program
pyFullPath=$programROOT/$pythonMain

# Set world size (total number of GPUs, or processes)
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_LOCALID

# Debugging info
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# Execute python code
python $pyFullPath --init_method tcp://$MASTER_ADDR:23111 --world_size $SLURM_NTASKS --config_file /nfs1/malla/Thesis/config.yaml
# END of SBATCH-script


## edit the config.yaml if you want to run train or test related job and then execute the below command
## sbatch slurm_script.sh

# to run tensorboard
# tensorboard --logdir='/nfs1/malla/Thesis/runs/path/here' --port=xxxx

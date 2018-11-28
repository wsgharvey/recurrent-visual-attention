#PBS -N "recurrent-visual-attention"
#PBS -q gpu
#PBS -M wsgh@cs.ubc.ca
#PBS -m abe

#set output and error directories
#PBS -o "localhost:/ubc/cs/home/w/wsgh/qsub-outputs/rmva.out"
#PBS -e "localhost:/ubc/cs/home/w/wsgh/qsub-outputs/rmva.err"
hostname

cd /ubc/cs/research/fwood/wsgh/attention/recurrent-visual-attention

python main.py --use_gpu True --use_tensorboard True --random_seed 1 --name notanh1

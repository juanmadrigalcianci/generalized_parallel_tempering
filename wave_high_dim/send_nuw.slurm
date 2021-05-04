#!/bin/bash

# First we set up the SLURM environment.
# All the SBATCH directives below are optional but they do
# allow you to modify the behaviour of your job.

#$number_of_processors=1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=juan.madrigalcianci@epfl.ch             # Where to send mail.                                                         # Linux/UNIX username or an email address
#SBATCH --verbose                                       # Increase informational messages
#SBATCH --partition=serial                                     # Increase informational messages
#SBATCH --job-name=GPT_Nuw
#SBATCH --ntasks=1			#$number_of_processors
#SBATCH --time=32:00:00


#SBATCH --mail-user=juan.madrigalcianci@epfl.ch
#SBATCH --mail-type=ALL
echo "I ran on:"
cd $SLURM_SUBMIT_DIR
echo $SLURM_NODELIST

module load intel
module load intel-mkl intel-mpi
#module load  boost cmake curl eigen hdf5 libtool metis mumps molden netcdf parmetis petsc scotch python


#gets run path
RUNPATH=/home/jmadriga/csqi-parallel_tempering_and_infinite_swapping/code_for_zenodo/generalized_parallel_tempering/wave
cd $RUNPATH


#loads Modules

source $HOME/anaconda3/bin/activate fenics

echo ==========================================


echo python version
echo ==========================================

which python

echo ==========================================
echo Loads modules
echo ==========================================
module load intel
module load intel-mkl intel-mpi
module load  boost cmake curl eigen hdf5 libtool metis mumps molden netcdf parmetis petsc scotch

echo ==========================================
echo Modules loaded
echo ==========================================

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/szip/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\$HOME/szip/lib
export IND=1234

srun $HOME/anaconda3/envs/fenics/bin/python ergodic_est_wave_nuw.py  $IND

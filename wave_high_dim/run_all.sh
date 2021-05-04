#!/bin/bash
# removes previous
# Does PT
line="$(grep -n "export IND=" send_fpt.slurm | head -n 1 | cut -d: -f1)"
for i in {102..152}
do

   #changes input line
   sed -i -e "${line}s/.*/export IND=$i/" send_fpt.slurm
   #submits the changed batch file, with IND as input
   sbatch send_fpt.slurm


done

# Does pCN

line="$(grep -n "export IND=" send_pcn.slurm | head -n 1 | cut -d: -f1)"
for i in {102..152}
do

   #changes input line
   sed -i -e "${line}s/.*/export IND=$i/" send_pcn.slurm
   #submits the changed batch file, with IND as input
   sbatch send_pcn.slurm


done


# Does SDPT

line="$(grep -n "export IND=" send_sd.slurm | head -n 1 | cut -d: -f1)"
for i in {102..152}
do

   #changes input line
   sed -i -e "${line}s/.*/export IND=$i/" send_sd.slurm
   #submits the changed batch file, with IND as input
   sbatch send_sd.slurm


done



# Does UW

line="$(grep -n "export IND=" send_uw.slurm | head -n 1 | cut -d: -f1)"
for i in {102..152}
do

   #changes input line
   sed -i -e "${line}s/.*/export IND=$i/" send_uw.slurm
   #submits the changed batch file, with IND as input
   sbatch send_uw.slurm


done


# Does W

line="$(grep -n "export IND=" send_w.slurm | head -n 1 | cut -d: -f1)"
for i in {102..152}
do

   #changes input line
   sed -i -e "${line}s/.*/export IND=$i/" send_w.slurm
   #submits the changed batch file, with IND as input
   sbatch send_w.slurm


done


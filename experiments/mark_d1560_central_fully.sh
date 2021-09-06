#!/bin/bash

exp_name="mark_d1560"

lmda_group="0.01 0.02 0.025 0.027 0.028 0.029 0.03 0.031 0.032 0.033 0.04 0.05 0.07"
N=385
d=1560
s=7
m=5
p=1
rho=0
solver_mode="centralized"
iter_type="lagrangian"
gamma=0.175

total=30

for lmda in $lmda_group
do
    num_exp=0
    echo $lmda
    while [ $num_exp -ne $total ]
    do
        command="-N $N -d $d -s $s -m $m -p $p -rho $rho --data_index $num_exp --solver_mode $solver_mode --iter_type $iter_type --gamma $gamma --lmda $lmda --verbose --storing_filepath ./output/$exp_name/N${N}_rho${rho}_exp${num_exp}/ --storing_filename ${solver_mode}_lambda${lmda}.output"
        python main.py $command &
        num_exp=$(($num_exp+1))
    done
    wait
done

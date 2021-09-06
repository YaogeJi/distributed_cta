#!/bin/bash

exp_name="gamma_veri_d1560"

N=385
d=1560
s=7
m=5
p=1
rho=0
solver_mode="distributed"
iter_type="lagrangian"
gamma_group="0.001 0.005 0.01 0.02 0.03 0.04 0.045 0.05 0.1 0.2 0.5"
total=30

for gamma in $gamma_group
do
    echo $gamma
    num_exp=0
    while [ $num_exp -ne $total ]
    do
        if [ "$num_exp" == "0" ]
        then
            command="-N $N -d $d -s $s -m $m -p $p -rho $rho --data_index $num_exp --solver_mode $solver_mode --iter_type $iter_type --gamma $gamma --optimal_lmda --verbose --storing_filepath ./output/$exp_name/N${N}_rho${rho}_exp${num_exp}/ --storing_filename ${solver_mode}_gamma${gamma}.output"
        else
            command="-N $N -d $d -s $s -m $m -p $p -rho $rho --data_index $num_exp --solver_mode $solver_mode --iter_type $iter_type --gamma $gamma --optimal_lmda --storing_filepath ./output/$exp_name/N${N}_rho${rho}_exp${num_exp}/ --storing_filename ${solver_mode}_gamma${gamma}.output"
        fi
        python main.py $command &
        num_exp=$(($num_exp+1))
    done
    wait
done

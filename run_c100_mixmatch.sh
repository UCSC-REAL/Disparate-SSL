pip install -r requirements.txt


mkdir result result_c100

mode="small ssl" 
numlabels="500" # 500 1000
numtimes="1 2 3 4 5"

for TIMES in $numtimes;
do
    for MMMODE in $mode;
    do
        for MMNUMLABEL in $numlabels;  
        do  
            python3 train.py --gpu 1 --n-labeled $MMNUMLABEL  --dataset cifar100 --sample Random --train_mode $MMMODE --out ./result > ./result_c100/mm-$MMMODE-$MMNUMLABEL-$TIMES.log
        done  
    done
done


python3 train.py --gpu 1 --n-labeled 44900  --dataset cifar100 --sample Random --train_mode small --out ./result > ./result_c100/mm-full-c100.log
pip install -r requirements.txt

mkdir result result_c10

mode="small ssl" 
numlabels="250" # 250 500
numtimes="1 2 3 4 5"

for TIMES in $numtimes;
do
    for MMMODE in $mode;
    do
        for MMNUMLABEL in $numlabels;  
        do  
            python3 train.py --gpu 0 --n-labeled $MMNUMLABEL  --dataset cifar10 --sample Random --train_mode $MMMODE --out ./result > ./result_c10/mm-$MMMODE-$MMNUMLABEL-$TIMES.log
        done  
    done
done


python3 train.py --gpu 0 --n-labeled 44900  --dataset cifar10 --sample Random --train_mode small --out ./result > ./result_c100/mm-full-c10.log

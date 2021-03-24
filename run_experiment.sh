
#!/usr/bin/env bash

#SEED=23
PATH_TO_DUMP='dump/results.json'

DATE=$(date +%H%M%S-%d%m)

mkdir -p ./dump


for obs_model in "range" "bearing" "range-bearing"; do
    for num_landmarks in 5 10 20 30; do
        for i in 1,1 2,3 3,5 4,7 5,10; do IFS=","; set -- $i;
            for seed in 101 102 103 104 105; do
                echo obs_model: ${obs_model} num_landmarks: ${num_landmarks} beta: $1,  $2 
                
                PYTHONPATH=. python3 src/main.py \
                    --obs_model ${obs_model} \
                    --num_landmarks ${num_landmarks} \
                    --beta $1 $2 \
                    --n_iter 250 \
                    --seed ${seed} \
                    --path_to_dump ${PATH_TO_DUMP} \
                    --verbose \
                    --plot_traj
            done        
        done
    done
done
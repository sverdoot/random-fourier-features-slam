
#!/usr/bin/env bash
defvalue=false
SEED=42
PATH_TO_DUMP='dump/results.json'
PATH_TO_TRAJS='data/trajs.npz'

DATE=$(date +%H%M%S-%d%m)

RM_RESULTS=${1:-$defvalue}

if [ $RM_RESULTS = true ] ; then
    if [ -f "$PATH_TO_DUMP" ] ; then
    rm "$PATH_TO_DUMP" && echo removed results.json
    fi
fi

mkdir -p ./dump

for obs_model in "range" "bearing" "range-bearing"; do
    for num_landmarks in 5 10 20 30; do
        for i in 1,1 2,3 3,5 4,7 5,10; do IFS=","; set -- $i;
            for traj_id in 0 1 6; do
                echo obs_model: ${obs_model} num_landmarks: ${num_landmarks} beta: $1,  $2 traj_id: ${traj_id}
                
                PYTHONPATH=. python3 src/main.py \
                    --obs_model ${obs_model} \
                    --num_landmarks ${num_landmarks} \
                    --beta $1 $2 \
                    --n_iter 100 \
                    --dampening_factor 5e-5 \
                    --seed ${SEED} \
                    --path_to_dump ${PATH_TO_DUMP} \
                    --path_to_trajs ${PATH_TO_TRAJS} \
                    --traj_id ${traj_id} \
                    --verbose \
                    --plot_traj
            done        
        done
    done
done
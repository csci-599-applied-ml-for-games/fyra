#!/bin/bash
# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
name=""
numseeds=2
retrain=""
new_env=""

while getopts "n:res:" opt; do
    case "$opt" in
    \?)
        echo "usage: docker_train.sh -n CONTAINER_NAME -s NUMBER_OF_SEEDS [-r RETRAIN=false] [-e NEW_ENV=false]"
        exit 0
        ;;
    n)  name=${OPTARG}
        ;;
    r)  retrain="_continue"
        ;;
    e)  new_env="--new_env"
        ;;
    s)  numseeds=${OPTARG}
        ;;
    esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

# generate run.sh
echo parallel python ./train_quad$retrain.py config/ppo__crazyflie_baseline.yml _results_temp/$name --seed {1} --nodisp $new_env ::: {1..$numseeds} > quad_train/run.sh 
echo results saved to _results_temp/$name

# docker commands
if [[ $name == "" ]]
then
  docker run -it -v $(pwd):/mount/quad_sim2multireal train
else
  docker run --name $name -it -v $(pwd):/mount/quad_sim2multireal train
fi


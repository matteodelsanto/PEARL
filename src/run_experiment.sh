#!/bin/bash 

# CUDA paths
#export PATH=/archive/home/mdelsant/cuda10.1_installation/bin:$PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/archive/home/mdelsant/cuda10.1_installation/lib64/
#export CUDA_HOME=/archive/home/mdelsant/cuda10.1_installation/
#export CUDA_PATH=/archive/home/mdelsant/cuda10.1_installation/

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/archive/home/mdelsant/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/archive/home/mdelsant/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/archive/home/mdelsant/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/archive/home/mdelsant/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


conda activate perplexityenv
cd /archive/home/mdelsant/PEARL/src/;python -m run.run_experiment

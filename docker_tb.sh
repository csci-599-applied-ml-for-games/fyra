#!/bin/bash

docker run -d -p 6006:6006 --name tensorboard -v $(pwd):/mount/quad_sim2multireal tb 

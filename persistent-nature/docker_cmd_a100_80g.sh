img="nvcr.io/nvidia/pytorch:22.05-py3" 

docker run --gpus all  --privileged=true   --workdir /git --name "persistent-nature"  -e DISPLAY --ipc=host -d --rm  -p 6332:8889 \
-v /raid/git/google-research/persistent-nature:/git/persistent-nature \
 -v /raid/git/datasets:/git/datasets \
 $img sleep infinity

docker exec -it persistent-nature /bin/bash

#docker images  |grep "pytorch"  |grep "21."

#docker stop  biobert

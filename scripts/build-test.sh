#!/bin/bash -x

set -e

#export TT_METAL_HOME=$(pwd)
#export TT_METAL_ENV=dev
#export ARCH_NAME=grayskull
#export PYTHONPATH=$(pwd)

#export DEVICE_CXX=/usr/bin/clang++

# Local hack, adding .pipcache under /work on my machine
[ -d /work/.pipcache ] || mkdir /work/.pipcache

function run_docker {
	sudo docker run \
		--rm -t -i \
		-v /work:/work \
		-v /dev/hugepages-1G:/dev/hugepages-1G \
		-w /work/tt-metal \
		-e TT_METAL_HOME=/work/tt-metal \
		-e TT_METAL_ENV=dev \
		-e ARCH_NAME=grayskull \
		-e PYTHONPATH=/work/tt-metal \
		-e XDG_CACHE_HOME=/work/.pipcache \
		-e DEVICE_CXX=/usr/bin/clang++ \
		--net host \
		--device /dev/tenstorrent/0 \
		--privileged \
		2004:latest \
		"$@"
}

run_docker make build -j$(nproc)
run_docker make tests -j$(nproc)

#. ./build/python_env/bin/activate
#pytest -svv models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference[LoFi-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_16]

#sudo docker run \
#	--rm -t -i \
#	-v /work:/work \
#	-v /dev/hugepages-1G:/dev/hugepages-1G \
#	-w /work/tt-metal \
#	-e TT_METAL_HOME=/work/tt-metal \
#	-e TT_METAL_ENV=dev \
#	-e ARCH_NAME=grayskull \
#	-e PYTHONPATH=/work/tt-metal \
#	-e XDG_CACHE_HOME=/work/.pipcache \
#	--device /dev/tenstorrent/0 \
#	--privileged \
#	2004:latest \
#	bash -c ". ./build/python_env/bin/activate ; pytest -rfs --junitxml=/work/tt-metal/junit.xml -svv 'models/demos/resnet/tests/test_perf_resnet.py::test_perf_virtual_machine[1-0.015-30]'"

run_docker bash -c ". ./build/python_env/bin/activate ; ./tests/scripts/run_tests.sh --tt-arch grayskull --pipeline-type models_performance_virtual_machine"

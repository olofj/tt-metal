FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils dialog
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libboost-all-dev sudo zip unzip
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tmux vim git
RUN apt-get install -y pciutils kmod python3-pip
RUN apt-get install -y python3.8 python3.8-dev python3.8-venv
RUN apt-get install -y curl jq
RUN apt-get install -y software-properties-common build-essential python3.8-venv libgoogle-glog-dev libyaml-cpp-dev libboost-all-dev libsndfile1 libhwloc-dev
RUN apt-get install -y clang-6.0 git git-lfs cmake pandoc libtbb-dev libcapstone-dev pkg-config
RUN apt-get install -y libgtest-dev
RUN apt-get install -y python-is-python3
RUN apt-get install -y clang
RUN useradd -m -s /bin/bash -u 1000 user

USER user



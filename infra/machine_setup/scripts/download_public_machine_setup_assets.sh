#!/usr/bin/env bash

set -eo pipefail

if [[ -z "$GITHUB_TOKEN" ]]; then
  echo "Must provide GITHUB_TOKEN in environment" 1>&2
  exit 1
fi

TT_METAL_HOME=$(git rev-parse --show-toplevel)
ASSETS_DIR=$TT_METAL_HOME/infra/machine_setup/assets

# FULL_REPO_NAME=tenstorrent-software/pybuda
FULL_REPO_NAME=tenstorrent-metal/tt-metal-sys-eng-packages

PYBUDA_GS_RELEASE=v1.0.23-tt-external.2f4901c8.gs
PYBUDA_WH_RELEASE=v1.0.24-tt-external.e67d8051.wh_b0
GS_TT_SMI_FILENAME=tt-smi_2022-12-05-74801e089fb2e564
WH_TT_SMI_FILENAME=tt-smi-wh-8.2.0.0_2023-04-19-621d61fca60a43eb
GS_TT_FLASH_FILENAME=tt-flash_2022-09-06-fae5785cae3807a6
WH_TT_FLASH_FILENAME=tt-flash_7.8.2.0_2023-03-29-ea858ffb9a5c19e3
GS_TT_DRIVER_FILENAME=ttkmd_1.14.tar.gz

PYBUDA_GS_RELEASE_ID=$(curl -L -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/vnd.github+json" \
	https://api.github.com/repos/$FULL_REPO_NAME/releases/tags/$PYBUDA_GS_RELEASE |
	jq '.id')

PYBUDA_GS_RELEASE_ASSETS=$(curl -L -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/vnd.github+json" \
	https://api.github.com/repos/$FULL_REPO_NAME/releases/$PYBUDA_GS_RELEASE_ID/assets)

GS_TT_SMI_SERVER_LOCATION=$(echo $PYBUDA_GS_RELEASE_ASSETS | jq ".[] | select(.name==\"$(echo $GS_TT_SMI_FILENAME)\")" | jq '.url' | tr \" \ )
# WH_TT_SMI_SERVER_LOCATION=$(echo $PYBUDA_GS_RELEASE_ASSETS | jq ".[] | select(.name==\"$(echo $WH_TT_SMI_FILENAME)\")" | jq '.url' | tr \" \ )
# GS_TT_FLASH_SERVER_LOCATION=$(echo $PYBUDA_GS_RELEASE_ASSETS | jq ".[] | select(.name==\"$(echo $GS_TT_FLASH_FILENAME)\")" | jq '.url' | tr \" \ )
# WH_TT_FLASH_SERVER_LOCATION=$(echo $PYBUDA_GS_RELEASE_ASSETS | jq ".[] | select(.name==\"$(echo $WH_TT_FLASH_FILENAME)\")" | jq '.url' | tr \" \ )
GS_TT_DRIVER_SERVER_LOCATION=$(echo $PYBUDA_GS_RELEASE_ASSETS | jq ".[] | select(.name==\"$(echo $GS_TT_DRIVER_FILENAME)\")" | jq '.url' | tr \" \ )

GS_TT_SMI_LOCAL_FOLDER=$ASSETS_DIR/tt_smi/grayskull
WH_TT_SMI_LOCAL_FOLDER=$ASSETS_DIR/tt_smi/wormhole_b0
GS_TT_FLASH_LOCAL_FOLDER=$ASSETS_DIR/tt_flash/grayskull
WH_TT_FLASH_LOCAL_FOLDER=$ASSETS_DIR/tt_flash/wormhole_b0
TT_DRIVER_LOCAL_FOLDER=$ASSETS_DIR/tt_driver

mkdir -p $GS_TT_SMI_LOCAL_FOLDER
mkdir -p $WH_TT_SMI_LOCAL_FOLDER
mkdir -p $GS_TT_FLASH_LOCAL_FOLDER
mkdir -p $WH_TT_FLASH_LOCAL_FOLDER
mkdir -p $TT_DRIVER_LOCAL_FOLDER

GS_TT_SMI_LOCAL_LOCATION=$GS_TT_SMI_LOCAL_FOLDER/tt-smi
WH_TT_SMI_LOCAL_LOCATION=$WH_TT_SMI_LOCAL_FOLDER/tt-smi
GS_TT_FLASH_LOCAL_LOCATION=$GS_TT_FLASH_LOCAL_FOLDER/tt-flash
WH_TT_FLASH_LOCAL_LOCATION=$WH_TT_FLASH_LOCAL_FOLDER/tt-flash
TT_DRIVER_LOCAL_LOCATION=$TT_DRIVER_LOCAL_FOLDER/ttkmd_1.14.tar.gz

echo $GS_TT_FLASH_LOCAL_LOCATION
echo $GS_TT_SMI_LOCAL_LOCATION
curl -vL -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/octet-stream" \
	$GS_TT_SMI_SERVER_LOCATION -o $GS_TT_SMI_LOCAL_LOCATION

curl -vL -H "Authorization: Bearer $GITHUB_TOKEN" \
	-H "X-GitHub-Api-Version: 2022-11-28" \
	-H "Accept: application/octet-stream" \
	$GS_TT_DRIVER_SERVER_LOCATION -o $TT_DRIVER_LOCAL_LOCATION

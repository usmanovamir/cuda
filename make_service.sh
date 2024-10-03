#!/bin/sh

GPU=$1
SK=$2

ADDR_HASH=739437bb3dd6d1983e66629c5f08c70e52769371

if [ -z $SK ]; then
    SK=`printf '%x' $(shuf -i 262144-524287 -n 1)`
fi

cat << EOF > brute_$GPU.service
[Unit]
Description=Brute $GPU
 
[Service]
Type=simple
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/brute ${SK}000000000000 ${ADDR_HASH} ${GPU}
MemoryMax=8G
User=root
 
[Install]
WantedBy=multi-user.target

EOF
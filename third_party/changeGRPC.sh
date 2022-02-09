#!/bin/sh
if [ -z "$1" ]; then
  echo "Usage: $0 GRPCPath"
  exit 1
fi
GRPC=$1
PROTUBUF="$1third_party/protobuf"
echo $PROTUBUF
cd $PROTUBUF 
#clean
make uninstall && make clean
#install
make && make install && ldconfig
cd /usr/local/lib
#rm libprotobuf-lite* && rm libprotobuf* && rm libprotoc*
#cd $PROTUBUF/src/.libs
#cp * /usr/local/lib
cd $GRPC
make clean
make && make install && ldconfig

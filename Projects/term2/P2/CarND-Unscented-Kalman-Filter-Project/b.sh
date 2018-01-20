WDIR=`pwd`
echo "ROOT_DIR: $WDIR"

# BUILD_MODE: [Debug/Release]
BUILD_MODE=$1

mkdir -p build/${BUILD_MODE} && cd build/${BUILD_MODE};
cmake \
    -DCMAKE_BUILD_TYPE=${BUILD_MODE} \
    $WDIR;

make -s;

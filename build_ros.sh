echo "Building ROS nodes"

cd Examples/ROS/orb_slam2
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j
cp devel/lib/orb_slam2/* ..

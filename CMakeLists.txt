project(OpenPose)

cmake_minimum_required(VERSION 2.6)
aux_source_directory(${PROJECT_SOURCE_DIR}/openpose DIR_SRCS)
add_definitions("-Wall -std=c++11")

include_directories(
	/usr/include/
	/usr/local/include
	/home/yfji/GitProjects/caffe/distribute/include
	/usr/local/cuda/include
	)
	
link_directories(
	/usr/lib
	/usr/local/lib/opencv_310
	/usr/lib/x86_64-linux-gnu
	/home/yfji/GitProjects/caffe/distribute/lib
	/usr/local/cuda/lib64
	)
	
add_library(openpose SHARED ${DIR_SRCS})

target_link_libraries(
	openpose
	opencv_core
	opencv_highgui
	opencv_imgproc
	opencv_imgcodecs
	opencv_videoio
	boost_python-py34
	boost_system
	boost_thread
	glog
	gflags
	lmdb
	leveldb
	hdf5
	hdf5_hl
	caffe
	)
	
add_executable(OpenPose ${DIR_SRCS})
target_link_libraries(
	OpenPose
	opencv_core
	opencv_highgui
	opencv_imgproc
	opencv_imgcodecs
	opencv_videoio
	boost_python-py34
	boost_system
	boost_thread
	glog
	protobuf
	gflags
	lmdb
	leveldb
	hdf5
	hdf5_hl
	caffe
	caffe
	)

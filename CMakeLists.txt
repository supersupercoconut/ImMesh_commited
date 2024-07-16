cmake_minimum_required(VERSION 2.8.3)
project(ImMesh)

SET(CMAKE_BUILD_TYPE "Release")

ADD_COMPILE_OPTIONS(-std=c++14 )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

if (ENABLE_ASAN)
    message(STATUS "build with ASAN")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -g")
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# 优化策略设置 O1 与 O3 为什么程序出现的错误是不一样的
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -g -O3 -ltbb -lboost_system -msse2 -msse3 -pthread -w")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -g -O3 -ltbb -lboost_system -msse2 -msse3 -pthread -w") # -Wall

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
    include(ProcessorCount)
    ProcessorCount(N)
    message("Processer number:  ${N}")
    if(N GREATER 5)
        add_definitions(-DMP_EN)
        add_definitions(-DMP_PROC_NUM=4)
        message("core for MP:  4")
    elseif(N GREATER 3)
        math(EXPR PROC_NUM "${N} - 2")
        add_definitions(-DMP_EN)
        add_definitions(-DMP_PROC_NUM="${PROC_NUM}")
        message("core for MP:  ${PROC_NUM}")
    else()
        add_definitions(-DMP_PROC_NUM=1)
    endif()
else()
    add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

# 这几个msgs有什么作用来着  geometry_msgs nav_msgs sensor_msgs std_msgs | 作者还是提供了两个msg格式

find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        nav_msgs
        sensor_msgs
        roscpp
        rospy
        std_msgs
        pcl_ros
        tf
        livox_ros_driver
        message_generation
        eigen_conversions
        cv_bridge
)


set(Glog_DIR /home/supercoconut/Myfile/thirdparty/glog-0.4.0/build)
find_package( Glog  REQUIRED PATHS /home/supercoconut/Myfile/thirdparty/glog-0.4.0 NO_DEFAULT_PATH)


find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CGAL REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem iostreams program_options system serialization)
set(Sophus_LIBRARIES libSophus.so)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        # ${OpenCV_INCLUDE_DIRS}
        ${Glog_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ./include

        # ===== Meshing =====
        ./include/ikd-Tree/
        ./src/meshing/
        ./src/meshing//r3live/
        ./src/meshing//kd_tree_dev/
        ./src/meshing//rgb_map/
        ./src/tools/
        ./src/tools/imgui
        ./src/tools/shader
        ./src/tools/openGL_libs
        ./src
)

add_message_files(
        FILES
        Pose6D.msg
        cloud_voxel.msg
        # States.msg
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        std_msgs
        nav_msgs
        sensor_msgs
)

catkin_package(
        CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
        DEPENDS EIGEN3 PCL OpenCV Sophus
        INCLUDE_DIRS
)

# set的作用是 设置变量值 —— 这里设置变量来代表路径 —— 并且使用add_definitions()来做宏定义 这样可以在代码中使用类似于#ifdef xx来判断是否定义了这个宏
set(FONTPATH ${CMAKE_CURRENT_SOURCE_DIR}/font/)
add_definitions(-DFONT_DIR=\"${FONTPATH}/\")

set(SHADERPATH ${CMAKE_CURRENT_SOURCE_DIR}/src/shader/)
add_definitions(-DSHADER_DIR=\"${SHADERPATH}/\")

add_executable(ImMesh_mapping
        src/ImMesh_node.cpp
        src/ImMesh_mesh_reconstruction.cpp
        include/ikd-Tree/ikd_Tree.cpp
        src/voxel_mapping.cpp
        src/voxel_mapping_common.cpp
        src/voxel_loc.cpp
        src/preprocess.cpp
        src/IMU_Processing.cpp

        ./src/meshing/mesh_rec_display.cpp
        ./src/meshing/mesh_rec_geometry.cpp
        ./src/meshing/r3live/triangle.cpp
        ./src/meshing/r3live/image_frame.cpp
        ./src/meshing/r3live/pointcloud_rgbd.cpp
        ./src/tools/openGL_libs/openGL_camera.cpp
        ./src/tools/openGL_libs/openGL_camera_view.cpp
        ./src/tools/openGL_libs/glad.c

        ./src/tools/imgui/imgui.cpp
        ./src/tools/imgui/imgui_draw.cpp
        ./src/tools/imgui/imgui_impl_glfw.cpp
        ./src/tools/imgui/imgui_impl_opengl3.cpp
        ./src/tools/imgui/imgui_tables.cpp
        ./src/tools/imgui/imgui_widgets.cpp
)

# tbb 并行处理的C++库 | 剩余的都不知道具体在哪个部分实现 | 没想到多线程的库竟然也要做库文件调用
target_link_libraries(ImMesh_mapping
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${PYTHON_LIBRARIES}
        CGAL::CGAL
        tbb
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${Boost_SYSTEM_LIBRARIES}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SERIALIZATION_LIBRARY} # serialization
        ${CERES_LIBRARIES}
        glog::glog
        pcl_common
        pcl_io
        pcl_filters
        pcl_kdtree
        GL GLU glfw X11 Xrandr pthread Xi dl Xinerama Xcursor)
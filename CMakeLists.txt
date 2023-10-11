cmake_minimum_required(VERSION 3.22)
project(TrainPhaseSplitWithTinyDNN)

add_executable(${PROJECT_NAME} main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
#target_compile_definitions(${PROJECT_NAME} PRIVATE CNN_USE_DOUBLE)

target_compile_options(${PROJECT_NAME} PRIVATE
	$<$<CXX_COMPILER_ID:MSVC>:/bigobj>
	$<$<CXX_COMPILER_ID:GNU>:-Wa,-mbig-obj>
)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
	thirdparty/Eigen3
	thirdparty/MiniDNN/include
)

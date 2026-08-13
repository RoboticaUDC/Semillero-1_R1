cmake_minimum_required(VERSION 3.18)
project(play_r1_isaac_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Torch ya NO es REQUIRED a nivel de proyecto -- solo se exige para play_r1_isaac
find_package(Torch QUIET)

set(MUJOCO_DIR "$ENV{HOME}/.mujoco/mujoco-3.3.6" CACHE PATH "Ruta a la instalacion de MuJoCo")
find_library(MUJOCO_LIB mujoco PATHS "${MUJOCO_DIR}/lib" REQUIRED)
find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h PATHS "${MUJOCO_DIR}/include" REQUIRED)

find_package(glfw3 REQUIRED)

# --- play_r1_isaac: solo se construye si Torch fue encontrado ---
if(Torch_FOUND)
    add_executable(play_r1_isaac play_r1_isaac.cpp)
    target_include_directories(play_r1_isaac BEFORE PRIVATE /usr/include "${MUJOCO_INCLUDE_DIR}")
    target_link_libraries(play_r1_isaac "${TORCH_LIBRARIES}" "${MUJOCO_LIB}" glfw pthread)
    set_target_properties(play_r1_isaac PROPERTIES
        BUILD_RPATH "${MUJOCO_DIR}/lib"
        INSTALL_RPATH "${MUJOCO_DIR}/lib")
else()
    message(WARNING "Torch no encontrado: se omite el target play_r1_isaac")
endif()

# --- teleop_r1: no depende de Torch ---
add_executable(teleop_r1 teleop_r1.cpp)
target_include_directories(teleop_r1 BEFORE PRIVATE /usr/include /usr/include/eigen3 "${MUJOCO_INCLUDE_DIR}")
target_link_libraries(teleop_r1 "${MUJOCO_LIB}" glfw pthread)
set_target_properties(teleop_r1 PROPERTIES
    BUILD_RPATH "${MUJOCO_DIR}/lib"
    INSTALL_RPATH "${MUJOCO_DIR}/lib")
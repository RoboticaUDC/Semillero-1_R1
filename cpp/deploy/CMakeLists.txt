cmake_minimum_required(VERSION 3.18)
project(r1_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LibTorch (cxx11 ABI):
#   cmake -DCMAKE_PREFIX_PATH=<ruta_torch> ..
find_package(Torch REQUIRED)

# unitree_sdk2 quedo instalado en /usr/local.
find_package(unitree_sdk2 REQUIRED)

add_executable(r1_deploy r1_deploy.cpp)

# IMPORTANTE: BEFORE fuerza que estas rutas se busquen ANTES que las que
# arrastra Torch (que incluye su propia copia de fmt/core.h, mas nueva que
# la del sistema, y choca con el libspdlog-dev de apt). Sin esto, el
# compilador puede resolver <fmt/...> contra la copia de torch en vez de
# /usr/include/fmt, rompiendo la compilacion de dds_wrapper (que usa spdlog).
target_include_directories(r1_deploy BEFORE PRIVATE
    /usr/include
    /usr/include/eigen3
    /usr/local/include/ddscxx
    /usr/local/include/iceoryx/v2.0.2
)

target_link_libraries(r1_deploy
    "${TORCH_LIBRARIES}"
    unitree_sdk2
    ddsc
    ddscxx
    spdlog
    fmt
    rt
    pthread
)
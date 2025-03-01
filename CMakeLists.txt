cmake_minimum_required(VERSION 3.10)
project(YoloVid CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Настройка для OpenCV через pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv4)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

# Настройка для MongoDB C++ Driver
include_directories(/usr/local/include/mongocxx/v_noabi /usr/local/include/bsoncxx/v_noabi)
link_directories(/usr/local/lib)

# Настройка для CURL
include_directories(/opt/homebrew/opt/curl/include)
link_directories(/opt/homebrew/opt/curl/lib)

# Настройка для nlohmann/json
include_directories(/opt/homebrew/opt/nlohmann-json/include)

# Добавляем исполняемый файл
add_executable(yolo_vid yolo_vid.cpp)

# Подключаем библиотеки
target_link_libraries(yolo_vid
    ${OpenCV_LIBRARIES}
    mongocxx
    bsoncxx
    curl
)
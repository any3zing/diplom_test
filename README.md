Компиляция: (cmake не работает)
"g++ -std=c++17 -O3 src/yolo_vid.cpp -o yolo_vid \
    `pkg-config --cflags --libs opencv4 ` \
    -I/usr/local/include/mongocxx/v_noabi \
    -I/usr/local/include/bsoncxx/v_noabi \
    -L/usr/local/lib \
    -lmongocxx -lbsoncxx  \
-I/opt/homebrew/opt/curl/include -L/opt/homebrew/opt/curl/lib -lcurl \
-I/opt/homebrew/opt/nlohmann-json/include"


Старт докера:
docker start mongodb
./yolo_example_vid



Вывод резов:
docker exec -it mongoldb mongosh
use yolo_database; 
db.detections.find();


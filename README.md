`mkdir build
cd build
cmake ..
make`

before starting ./yolo_vid
you should 
1) run docker
`docker start mongodb`
3) run server 
`python3 server.py`


Moreover you can watch whats inside monngo by

`docker exec -it mongoldb mongosh
use yolo_database; 
db.detections.find();`

![Uploading Снимок экрана 2025-03-01 в 20.17.31.png…]()

*Make sure you have already on your system*:

1. Any modern UNIX OS (tested on Monterey 12.12.1)
2. OpenCV 4.5.4+
3. GCC 11.0+
IMPORTANT!!! Note that OpenCV versions prior to 4.5.4 will not work at all.

`git clone https://github.com/any3zing/diplom_test.git`
`cd diplom_test`

Upload your model file (.onnx) to config_files.

*BUILD*

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

![alt text](https://github.com/any3zing/diplom_test/blob/12112e21796cb8119584f374f2a87ddbc895ae48/result.png)

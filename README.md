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

[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) lh-tool
===

Kserve模型推理服务demo

# 安装

在GPU服务器克隆代码
```shell
git clone https://github.com/lh9171338/Kserve-Demo.git
```

安装依赖库
```shell
cd Kserve-Demo
python -m pip install -r requirements.txt
```

# 启动服务

在GPU服务器启动服务

```shell
python server.py
```

在服务器端测试服务请求
```shell
python service.py
```

输入如下：
```shell
airplane.jpg: {'score': 0.9807844161987305, 'label': 404, 'class_name': 'airliner'}
car.jpg: {'score': 0.9351834058761597, 'label': 817, 'class_name': 'sports car, sport car'}
cat.jpg: {'score': 0.40157631039619446, 'label': 282, 'class_name': 'tiger cat'}
dog.jpg: {'score': 0.45859256386756897, 'label': 162, 'class_name': 'beagle'}

```

# 请求服务

在本地请求服务(将`service.py`和`images`拷贝到本地)
```shell
python service.py --ip <server ip>
```

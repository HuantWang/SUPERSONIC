# Acquiring gRPC
We use gRPC to connected our tvm-mcts system, so we have to build the gRPC first. The version of gRPC we used is v1.12.0.

```shell
# 查看版本
$ protoc --version
libprotoc 3.5.1
$ pip l
protobuf                      3.16.0
```



# adding this file to 
```sh
(compiler_gym) root@0be5e588a6c8:site-packages# pwd
/root/anaconda3/envs/compiler_gym/lib/python3.8/site-packages
(compiler_gym) root@0be5e588a6c8:site-packages# cat facebook.pth 
<path>
<path>/grpc_tvm
```



## 利用grpc工具, 将protoc文件生成对应的头文件

```shell
// 生成python头文件的命令
// python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. schedule.proto
// 生成c++头文件的命令 服务类和消息类
// protoc -I ./ --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` schedule.proto
// protoc -I ./ --cpp_out=. schedule.proto
```

# 测试


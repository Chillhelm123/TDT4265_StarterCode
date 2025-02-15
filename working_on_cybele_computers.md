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
python train.py --workers 8 --device 0 --batch-size 32 --data data/RDD2022.yaml --img 640 640 --cfg cfg/training/yolov7-e6e-RDD2022.yaml --weights 'yolov7-e6e_training.pt' --name yolov7-e6e-RDD2022 --hyp data/hyp.scratch.custom.yaml
python train_aux.py --workers 8 --device 0 --batch-size 8 --data data/RDD2022.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e-RDD2022.yaml --weights 'yolov7-e6e_training.pt' --name yolov7-e6e-RDD2022 --hyp data/hyp.scratch.custom.yaml

# Working on Cybele Computers

There are 25 computers in Cybele (the lab previously known as Tulipan), all with powerful GPU cards (1080, 1080ti etc). There are a lot of students in the class and you have to consider your fellow students when you are using the GPU resources.

We ask you to follow these rules when using the computers:

1. Each group can only use a single computer at a time.
2. It is not allowed to remote access the computers in school time. This is the time 08:00-20:00 every weekday (Monday-Friday).
3. Before you start utilizing the GPU, check that no one is using it with the command `nvidia-smi`.


## Environment
Every computer in the cybele lab comes with python2 and python3.
You can run code in python3 by using, `python3 my_program.py`.


### Installing packages
If you want to install additional packages in your environment, install it locally. For example

```bash
pip3 install --user package_name
```

To get a list of already installed packages, use:
```
pip list
```

Pytorch is already installed on the computers and it should work out of the box with the GPU. Just launch python with "python3" in the terminal.


## Working remote
You can access the computer from your home computer/laptop by using ssh. To connect, you can connect to the following adress:

```
ssh [ntnu-username]@clab[00-25].idi.ntnu.no
```
For example, 
```
ssh mamoonas@clab21.idi.ntnu.no
```
You need to be on the school network to be able to connect.
If you want to connect outside school network, you have to use a VPN. \href{https://innsida.ntnu.no/wiki/-/wiki/English/Install+vpn}{(Innsida VPN guide).}

## Using Jupyter remotely
You can work in jupyter notebook remotely from your home using ssh port forwarding:

```
ssh -o "ServerAliveInterval=60" -L 7050:localhost:7050 mamoonas@clab01.idi.ntnu.no
```
You can use any available local port instead of 7050. Then open jupyter:
```
jupyter notebook --no-browser --port=7050 
```
Then copy paste one of the given URLs in your browser to open jupyter.

**Same can be done on IDUN cluster!**

## Selecting CUDA version
You can select which cuda version to use (version 9 or 11), by issuing the following cmd;
```
cuda-ver 9.0
```
**NOTE**: I recommend installing conda. 


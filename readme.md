# Facial Expression Recognition in facial occlusion scenarios: a Path Selection Multi-network

![image-20220608194550274](readme.assets/image-20220608194550274-16551991378131.png)

## Code

### Install dependencies

```python 3.6.13
pytorch 1.10.2 (1.5/1.6)
opencv-python 4.4.0.42
numpy 1.19.2
pandas 1.1.5
tqdm
```

### Prepare datasets 

We take four open-source emotion datasets: RAF-DB, Fer2013, Jaffe, KDEF. We need to convert them into .csv for future creation of datasets.

```
python jaffe2csv.py
python kdef2csv.py
python rafdb2csv.py
```



![image-20220608195712139](readme.assets/image-20220608195712139-16551991424872.png)

### The Creation of Datasets

```python createDatabase.py```

After this, we can get ConcatDB, BConcatDB, eyeCDB. Then these dataset folder need to be arranged in the following form before training. For example, beginConcatDB is used to train BeginNet and sepConcatDB is used to train Subnets.

```
beginConcatDB/beginBConcatDB/begineyeCDB
|--Train
	|--0
	|--1
	|--2
|--Val
	|--0
	|--1
	|--2
|--Test
	|--0happy
	|--1normal
	|--2sad
	|--3anger
	|--4fear
	|--5disgust
	|--6surprise

sepConcatDB/sepBConcatDB/sepeyeCDB
|--Train
	|--0
		|--0happy
		|--1normal
		|--2sad
	|--1
		|--3anger
		|--4fear
	|--2
		|--5disgust
		|--6surprise
|--Val
	|--0
		|--0happy
		|--1normal
		|--2sad
	|--1
		|--3anger
		|--4fear
	|--2
		|--5disgust
		|--6surprise
|--Test
	|--0happy
	|--1normal
	|--2sad
	|--3anger
	|--4fear
	|--5disgust
	|--6surprise

```

Using the following Commands. We also take ConcatDB as an example.

```
cp -r ConcatDB beginConcatDB
cp -r ConcatDB sepConcatDB
cd beginConcatDB
cp -r 0happy/. 0
cp -r 1normal/. 0
cp -r 2sad/. 0
cp -r 3anger/. 1
cp -r 4fear/. 1
cp -r 5disgust/. 2
cp -r 6surprised/. 2
rm -r 0happy
rm -r 1normal
rm -r 2sad
rm -r 3anger
rm -r 4fear
rm -r 5disgust
rm -r 6surprised

cd sepConcatDB
mkdir 0
mkdir 1
mkdir 2
mkdir 0/0happy
mkdir 0/1normal
mkdir 0/2sad
mkdir 1/3anger
mkdir 1/4fear
mkdir 2/5disgust
mkdir 2/6surprised

cp -r 0happy/. 0/0happy
cp -r 1normal/. 0/1normal
cp -r 2sad/. 0/2sad
cp -r 3anger/. 1/3anger
cp -r 4fear/. 1/4fear
cp -r 5disgust/. 2/5disgust
cp -r 6surprised/. 2/6surprised
rm -r 0happy
rm -r 1normal
rm -r 2sad
rm -r 3anger
rm -r 4fear
rm -r 5disgust
rm -r 6surprised
```

### Train BeginNet

Before training a model, edit `config.py` with desired parameters.

--data_dir dataset used to train beginNet

--scenarios which occlusion scenario is this dataset indicate

```python trainBeginNet.py --data_dir beginConcatDB --scenarios upper```

### Train Subnets

```python trainSubnets.py --data_dir sepConcatDB --scenarios upper```

After training, the SubnetX.pth, SubnetY.pth, SubnetZ.pth will be saved together in ModelFolder

e.g

``` beginConcatDB/beginBConcatDB/begineyeCDB
./models/Subnets/upper/resnext50_1654343507.6094346
	|--SubnetX
		|--SubnetX.pth
		|--some records during training
	|--SubnetY
		|--SubnetY.pth
	|--SubnetZ
		|--SubnetZ.pth
```

Sometimes the weights of SubnetY won't be updated during training, so we may have to train SubnetY alone using trainSubnetY.py. Remember to put the trained SubnetY model (the .pth file) together with SubnetX and SubnetZ models. 

### Test

```python test.py --data_dir ConcatDB --SubnetsPath ./models/Subnets/upper/ModelFolder --beginNetPath ./models/BeginNet/upper/ModelFolder```

### Predict Real-World images

Put the real-world images you want to predict in imageFolder (all in the same occlusion scenario).

e.g All in the upper face occlusion scenario

```python PredictImg.py --img_path imageFolder --SubnetsPath ./models/Subnets/upper/ModelFolder --beginNetPath ./models/BeginNet/upper/ModelFolder```

### Related Paper

Liheng Ruan, Yuexing Han*, Jiarui Sun, Qiaochuan Chen, Jiaqi Li. Facial expression recognition in facial occlusion scenarios: A path selection multi-network. Displays, DOI: https://doi.org/10.1016/j.displa.2022.102245,  S0141-9382(22)00070-1, 2022. 
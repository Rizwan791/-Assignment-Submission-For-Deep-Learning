# -Assignment-Submission-For-Deep-Learning
 Assignment Submission For Deep Learning 
 

Car Price Prediction:: Download dataset from this link:
https://www.kaggle.com/hellbuoy/car-price-prediction
Problem Statement:: A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.
They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:
Which variables are significant in predicting the price of a car How well those variables describe the price of a car Based on various market surveys, the consulting firm has gathered a large data set of different types of cars across the America market.
task:: We are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market.
WORKFLOW :: 1.Load Data
2.Check Missing Values ( If Exist ; Fill each record with mean of its feature )
3.Split into 50% Training(Samples,Labels) , 30% Test(Samples,Labels) and 20% Validation Data(Samples,Labels).
4.Model : input Layer (No. of features ), 3 hidden layers including 10,8,6 unit & Output Layer with activation function relu/tanh (check by experiment).
5.Compilation Step (Note : Its a Regression problem , select loss , metrics according to it) 6.Train the Model with Epochs (100) and validate it
7.If the model gets overfit tune your model by changing the units , No. of layers , activation function , epochs , add dropout layer or add Regularizer according to the need .
8.Evaluation Step
9.Prediction
In [5]:
import numpy as np 
import pandas as pd
In [6]:
from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
In [9]:
data = pd.read_csv("/content/drive/MyDrive/CarPrice_Assignment.csv")
In [10]:
data.head()
Out[10]:
car_ID	symboling	CarName	fueltype	aspiration	doornumber	carbody	drivewheel	enginelocation	wheelbase	carlength	carwidth	carheight	curbweight	enginetype	cylindernumber	enginesize	fuelsystem	boreratio	stroke	compressionratio	horsepower	peakrpm	citympg	highwaympg	price
0	1	3	alfa-romero giulia	gas	std	two	convertible	rwd	front	88.6	168.8	64.1	48.8	2548	dohc	four	130	mpfi	3.47	2.68	9.0	111	5000	21	27	13495.0
1	2	3	alfa-romero stelvio	gas	std	two	convertible	rwd	front	88.6	168.8	64.1	48.8	2548	dohc	four	130	mpfi	3.47	2.68	9.0	111	5000	21	27	16500.0
2	3	1	alfa-romero Quadrifoglio	gas	std	two	hatchback	rwd	front	94.5	171.2	65.5	52.4	2823	ohcv	six	152	mpfi	2.68	3.47	9.0	154	5000	19	26	16500.0
3	4	2	audi 100 ls	gas	std	four	sedan	fwd	front	99.8	176.6	66.2	54.3	2337	ohc	four	109	mpfi	3.19	3.40	10.0	102	5500	24	30	13950.0
4	5	2	audi 100ls	gas	std	four	sedan	4wd	front	99.4	176.6	66.4	54.3	2824	ohc	five	136	mpfi	3.19	3.40	8.0	115	5500	18	22	17450.0
In [11]:
data.isnull().sum()
Out[11]:
car_ID              0
symboling           0
CarName             0
fueltype            0
aspiration          0
doornumber          0
carbody             0
drivewheel          0
enginelocation      0
wheelbase           0
carlength           0
carwidth            0
carheight           0
curbweight          0
enginetype          0
cylindernumber      0
enginesize          0
fuelsystem          0
boreratio           0
stroke              0
compressionratio    0
horsepower          0
peakrpm             0
citympg             0
highwaympg          0
price               0
dtype: int64
In [12]:
car_price.isna().sum()
Out[12]:
car_ID              0
symboling           0
CarName             0
fueltype            0
aspiration          0
doornumber          0
carbody             0
drivewheel          0
enginelocation      0
wheelbase           0
carlength           0
carwidth            0
carheight           0
curbweight          0
enginetype          0
cylindernumber      0
enginesize          0
fuelsystem          0
boreratio           0
stroke              0
compressionratio    0
horsepower          0
peakrpm             0
citympg             0
highwaympg          0
price               0
dtype: int64
In [14]:
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 205 entries, 0 to 204
Data columns (total 26 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   car_ID            205 non-null    int64  
 1   symboling         205 non-null    int64  
 2   CarName           205 non-null    object 
 3   fueltype          205 non-null    object 
 4   aspiration        205 non-null    object 
 5   doornumber        205 non-null    object 
 6   carbody           205 non-null    object 
 7   drivewheel        205 non-null    object 
 8   enginelocation    205 non-null    object 
 9   wheelbase         205 non-null    float64
 10  carlength         205 non-null    float64
 11  carwidth          205 non-null    float64
 12  carheight         205 non-null    float64
 13  curbweight        205 non-null    int64  
 14  enginetype        205 non-null    object 
 15  cylindernumber    205 non-null    object 
 16  enginesize        205 non-null    int64  
 17  fuelsystem        205 non-null    object 
 18  boreratio         205 non-null    float64
 19  stroke            205 non-null    float64
 20  compressionratio  205 non-null    float64
 21  horsepower        205 non-null    int64  
 22  peakrpm           205 non-null    int64  
 23  citympg           205 non-null    int64  
 24  highwaympg        205 non-null    int64  
 25  price             205 non-null    float64
dtypes: float64(8), int64(8), object(10)
memory usage: 41.8+ KB
In [16]:
data.head()
Out[16]:
car_ID	symboling	CarName	fueltype	aspiration	doornumber	carbody	drivewheel	enginelocation	wheelbase	carlength	carwidth	carheight	curbweight	enginetype	cylindernumber	enginesize	fuelsystem	boreratio	stroke	compressionratio	horsepower	peakrpm	citympg	highwaympg	price
0	1	3	alfa-romero giulia	gas	std	two	convertible	rwd	front	88.6	168.8	64.1	48.8	2548	dohc	four	130	mpfi	3.47	2.68	9.0	111	5000	21	27	13495.0
1	2	3	alfa-romero stelvio	gas	std	two	convertible	rwd	front	88.6	168.8	64.1	48.8	2548	dohc	four	130	mpfi	3.47	2.68	9.0	111	5000	21	27	16500.0
2	3	1	alfa-romero Quadrifoglio	gas	std	two	hatchback	rwd	front	94.5	171.2	65.5	52.4	2823	ohcv	six	152	mpfi	2.68	3.47	9.0	154	5000	19	26	16500.0
3	4	2	audi 100 ls	gas	std	four	sedan	fwd	front	99.8	176.6	66.2	54.3	2337	ohc	four	109	mpfi	3.19	3.40	10.0	102	5500	24	30	13950.0
4	5	2	audi 100ls	gas	std	four	sedan	4wd	front	99.4	176.6	66.4	54.3	2824	ohc	five	136	mpfi	3.19	3.40	8.0	115	5500	18	22	17450.0
In [30]:
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
data["fueltype"]=labelencoder.fit_transform(data["fueltype"])
data["aspiration"]=labelencoder.fit_transform(data["aspiration"])
data["carbody"]=labelencoder.fit_transform(data["carbody"])
data["drivewheel"]=labelencoder.fit_transform(data["drivewheel"])
data["enginetype"]=labelencoder.fit_transform(data["enginetype"])
data["cylindernumber"]=labelencoder.fit_transform(data["cylindernumber"])
data["enginelocation"]=labelencoder.fit_transform(data["enginelocation"])
data["CarName"]=labelencoder.fit_transform(data["CarName"])
data["doornumber"]=labelencoder.fit_transform(data["doornumber"])
data["fuelsystem"]=labelencoder.fit_transform(data["fuelsystem"])
In [32]:
data.head()
Out[32]:
car_ID	symboling	CarName	fueltype	aspiration	doornumber	carbody	drivewheel	enginelocation	wheelbase	carlength	carwidth	carheight	curbweight	enginetype	cylindernumber	enginesize	fuelsystem	boreratio	stroke	compressionratio	horsepower	peakrpm	citympg	highwaympg	price
0	1	3	2	1	0	1	0	2	0	88.6	168.8	64.1	48.8	2548	0	2	130	5	3.47	2.68	9.0	111	5000	21	27	13495.0
1	2	3	3	1	0	1	0	2	0	88.6	168.8	64.1	48.8	2548	0	2	130	5	3.47	2.68	9.0	111	5000	21	27	16500.0
2	3	1	1	1	0	1	2	2	0	94.5	171.2	65.5	52.4	2823	5	3	152	5	2.68	3.47	9.0	154	5000	19	26	16500.0
3	4	2	4	1	0	0	3	1	0	99.8	176.6	66.2	54.3	2337	3	2	109	5	3.19	3.40	10.0	102	5500	24	30	13950.0
4	5	2	5	1	0	0	3	0	0	99.4	176.6	66.4	54.3	2824	3	1	136	5	3.19	3.40	8.0	115	5500	18	22	17450.0
In [41]:
mean = data[["wheelbase","carlength",	"carwidth"	,"carheight"	,"curbweight" ,"enginesize" , "boreratio"	,"stroke",	"compressionratio",	"horsepower",	"peakrpm",	"citympg"	,"highwaympg"]].mean(axis=0)
data[["wheelbase","carlength",	"carwidth"	,"carheight"	,"curbweight" ,"enginesize" , "boreratio"	,"stroke",	"compressionratio",	"horsepower",	"peakrpm",	"citympg"	,"highwaympg"]] -= mean
std = data[["wheelbase","carlength",	"carwidth"	,"carheight"	,"curbweight" ,"enginesize" , "boreratio"	,"stroke",	"compressionratio",	"horsepower",	"peakrpm",	"citympg"	,"highwaympg"]].std(axis=0)
data[["wheelbase","carlength",	"carwidth"	,"carheight"	,"curbweight" ,"enginesize" , "boreratio"	,"stroke",	"compressionratio",	"horsepower",	"peakrpm",	"citympg"	,"highwaympg"]] /= std
data.head()
Out[41]:
car_ID	symboling	CarName	fueltype	aspiration	doornumber	carbody	drivewheel	enginelocation	wheelbase	carlength	carwidth	carheight	curbweight	enginetype	cylindernumber	enginesize	fuelsystem	boreratio	stroke	compressionratio	horsepower	peakrpm	citympg	highwaympg	price
0	1	3	2	1	0	1	0	2	0	-1.686643	-0.425480	-0.842719	-2.015483	-0.014531	0	2	0.074267	5	0.517804	-1.834886	-0.287645	0.174057	-0.262318	-0.644974	-0.544725	13495.0
1	2	3	3	1	0	1	0	2	0	-1.686643	-0.425480	-0.842719	-2.015483	-0.014531	0	2	0.074267	5	0.517804	-1.834886	-0.287645	0.174057	-0.262318	-0.644974	-0.544725	16500.0
2	3	1	1	1	0	1	2	2	0	-0.706865	-0.230948	-0.190101	-0.542200	0.513625	5	3	0.602571	5	-2.399008	0.684271	-0.287645	1.261448	-0.262318	-0.950684	-0.689938	16500.0
3	4	2	4	1	0	0	3	1	0	0.173274	0.206750	0.136209	0.235366	-0.419770	3	2	-0.430023	5	-0.516003	0.461055	-0.035885	-0.053537	0.785932	-0.186409	-0.109087	13950.0
4	5	2	5	1	0	0	3	0	0	0.106848	0.206750	0.229440	0.235366	0.515545	3	1	0.218350	5	-0.516003	0.461055	-0.539405	0.275209	0.785932	-1.103540	-1.270789	17450.0
In [43]:
a =data.corr()
import matplotlib.pyplot as plt

a.plot(kind="bar")
Out[43]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f6d93b3d7d0>

In [49]:
data.head()
Out[49]:
car_ID	symboling	CarName	fueltype	aspiration	doornumber	carbody	drivewheel	enginelocation	wheelbase	carlength	carwidth	carheight	curbweight	enginetype	cylindernumber	enginesize	fuelsystem	boreratio	stroke	compressionratio	horsepower	peakrpm	citympg	highwaympg	price
0	1	3	2	1	0	1	0	2	0	-1.686643	-0.425480	-0.842719	-2.015483	-0.014531	0	2	0.074267	5	0.517804	-1.834886	-0.287645	0.174057	-0.262318	-0.644974	-0.544725	13495.0
1	2	3	3	1	0	1	0	2	0	-1.686643	-0.425480	-0.842719	-2.015483	-0.014531	0	2	0.074267	5	0.517804	-1.834886	-0.287645	0.174057	-0.262318	-0.644974	-0.544725	16500.0
2	3	1	1	1	0	1	2	2	0	-0.706865	-0.230948	-0.190101	-0.542200	0.513625	5	3	0.602571	5	-2.399008	0.684271	-0.287645	1.261448	-0.262318	-0.950684	-0.689938	16500.0
3	4	2	4	1	0	0	3	1	0	0.173274	0.206750	0.136209	0.235366	-0.419770	3	2	-0.430023	5	-0.516003	0.461055	-0.035885	-0.053537	0.785932	-0.186409	-0.109087	13950.0
4	5	2	5	1	0	0	3	0	0	0.106848	0.206750	0.229440	0.235366	0.515545	3	1	0.218350	5	-0.516003	0.461055	-0.539405	0.275209	0.785932	-1.103540	-1.270789	17450.0
In [51]:
data.drop(["car_ID"],axis=1)
Out[51]:
symboling	CarName	fueltype	aspiration	doornumber	carbody	drivewheel	enginelocation	wheelbase	carlength	carwidth	carheight	curbweight	enginetype	cylindernumber	enginesize	fuelsystem	boreratio	stroke	compressionratio	horsepower	peakrpm	citympg	highwaympg	price
0	3	2	1	0	1	0	2	0	-1.686643	-0.425480	-0.842719	-2.015483	-0.014531	0	2	0.074267	5	0.517804	-1.834886	-0.287645	0.174057	-0.262318	-0.644974	-0.544725	13495.0
1	3	3	1	0	1	0	2	0	-1.686643	-0.425480	-0.842719	-2.015483	-0.014531	0	2	0.074267	5	0.517804	-1.834886	-0.287645	0.174057	-0.262318	-0.644974	-0.544725	16500.0
2	1	1	1	0	1	2	2	0	-0.706865	-0.230948	-0.190101	-0.542200	0.513625	5	3	0.602571	5	-2.399008	0.684271	-0.287645	1.261448	-0.262318	-0.950684	-0.689938	16500.0
3	2	4	1	0	0	3	1	0	0.173274	0.206750	0.136209	0.235366	-0.419770	3	2	-0.430023	5	-0.516003	0.461055	-0.035885	-0.053537	0.785932	-0.186409	-0.109087	13950.0
4	2	5	1	0	0	3	0	0	0.106848	0.206750	0.229440	0.235366	0.515545	3	1	0.218350	5	-0.516003	0.461055	-0.539405	0.275209	0.785932	-1.103540	-1.270789	17450.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
200	-1	139	1	0	0	3	2	0	1.717669	1.195622	1.394830	0.726460	0.761377	3	2	0.338419	5	1.662375	-0.336147	-0.161765	0.249921	0.576282	-0.339264	-0.399512	16845.0
201	-1	138	1	1	0	3	2	0	1.717669	1.195622	1.348215	0.726460	0.947672	3	2	0.338419	5	1.662375	-0.336147	-0.363173	1.413178	0.366632	-0.950684	-0.835151	19045.0
202	-1	140	1	0	0	3	2	0	1.717669	1.195622	1.394830	0.726460	0.876611	5	3	1.106861	5	0.923942	-1.229012	-0.337997	0.755685	0.785932	-1.103540	-1.125577	21485.0
203	-1	142	0	1	0	3	2	0	1.717669	1.195622	1.394830	0.726460	1.270327	3	3	0.434474	3	-1.180593	0.461055	3.236992	0.047616	-0.681618	0.119302	-0.544725	22470.0
204	-1	143	1	1	0	3	2	0	1.717669	1.195622	1.394830	0.726460	0.972640	3	2	0.338419	5	1.662375	-0.336147	-0.161765	0.249921	0.576282	-0.950684	-0.835151	22625.0
205 rows × 25 columns
In [52]:
X = data.loc[:,"symboling":"highwaympg"]
In [54]:
Y=data["price"]
In [57]:
Y
Out[57]:
0      13495.0
1      16500.0
2      16500.0
3      13950.0
4      17450.0
        ...   
200    16845.0
201    19045.0
202    21485.0
203    22470.0
204    22625.0
Name: price, Length: 205, dtype: float64
In [ ]:

# C4BDC-酥脆饼干
我们今年参加的是：第五届中国高校计算机大赛——华为云大数据挑战赛。为了不断作死的跳出舒适圈，进军更有挑战性的大数据和人工智能领域，作为气候爱好者的本人诚邀了同为苏黎世联邦理工数据科学专业的福娃和机器人专业的嗷嗷一块组成酥脆饼干队刷怪上分。奈何初赛一开始咱们全队都在实习，都没啥时间整baseline(代码模板)。搞得非要离初赛结束10天左右才有机会提交结果。要不是芜湖航空队的姜大佬给众多比赛队提供baseline，我们自己设计一个，感觉就是暗无天日了！


我们的上分之路非常的坎坷，从27万，12万，6万，4万。。。尽管我们队每天的分数是上一天的一半左右，仍然与6月22号开始的1万开始算起，半衰期为3天左右的复赛线相距甚远。咱们的初赛分A榜准备榜和B榜决胜榜，A榜的时候到了6100分的时候真的是没辙了，运算量大到暂时没法上分了，无奈由于时间关系只能转战B榜。简单来说，要提交一次结果，首先得让电脑去训练指定数据(简单来说分为：数据清洗、特征提取(训练集和测试集)、建模、整理输出格式)
​

感谢咱们的二位同伴清洗数据和提取数据特征，要不咱们这个队真的就没啥前进的动力了。。。B榜的时候我们换了一个思路，训练数据的时候直接把一个卫星追踪点为单位的特征转换成一个运单号为单位的特征(要知道一个运单号包含了上千或上万个卫星追踪点)，这样不但大大提高了模型训练的效率，选对了特征以后模型拟合的准确率也大幅提高。这也是为什么咱们队能从B榜的6万多分一下子上到2200多分，并且名次在初赛B榜期间会一次性跃升142名，进到70多名，并且最终也苟住并且进了复赛圈。


那现在就是技术环节了：(Python3的环境)

Part1. 需要的包
```Python
from tqdm import tqdm
import numpy as np
import math

from sklearn.model_selection import KFold

# lightgbm包比Xgboost有效率的多，cpu占用和内存占用都比较少，因此
# lightgbm就成了初赛的标配
import lightgbm as lgb

import os
import glob

# 屏蔽不必要的警报讯息
import warnings
warnings.filterwarnings('ignore')
```

Part2. 读取数据
```Python
# baseline只用到gps定位数据，即train_gps_path
train_gps_path = xxx
test_data_path = xxx
order_data_path = xxx
port_data_path = xxx

# 获取数据：把数据类型重新整理一下，尤其整理时间戳数据
def get_data(): 
  xxx
  
# 读取测试集，按航线来分类，为后面分解测试集数据做铺垫
test_data = pd.read_csv('你的路径'+test_data_path)
course_list = list(test_data["TRANSPORT_TRACE"].unique())

# 运单号的数量和每个运单号的数据长度
order_list = test_data['loadingOrder'].unique()
order_length = []
for i in range(len(order_list)):
    dff = test_data[test_data['loadingOrder'] == order_list[i]].shape[0]
    order_length.append(dff)
```
Part3. 清洗数据

```Python
# order_list指的是训练集中能够预测测试集指定路线的运单号
def data_process(df, order_list, mode='train'):
    assert mode=='train' or mode=='test'
    if mode == 'train':
        df = df.drop_duplicates(keep='first')
        df = df[df['loadingOrder'].isin(order_list)]
    return df

# 搞不好要定义两种数据清洗的函数
# (让服务于测试集的数据行数变少到原来的14%左右)
# 一个拿来去重，把新的训练集保存成按测试集路线分的文件
def train_process1():
  # 这里我们因为要读取上亿行数据，所以采用分块读取的策略
  # 每块表格的行数为：NDATA
  reader = pd.csv_read(文件名, iterator = TRUE,)
  chunk_cnt = 0
  while True: 
    try: 
      df = reader.get_chunk()
      # 这里就对数据进行清理，并且新数据和老数据合并去重
      chunk_cnt += 1
    except: 
      break 
# 中间有一部分清洗数据是咱们队员贡献的，主要是用来筛掉天数小于4的订单，
# 和速度超过50km/h的订单(这不是货船的速度)

# 另一个拿来直接进一步洗按测试集路线分的文件
def train_process2():
  xxx
```
Part4. 抓取特征(这个任务主要是福娃做的)
```Python
# 抓取特征
## 特征简介

- 把经纬度和位移转换成xyz坐标和球面最短曲线
- 把船开的方向改成xyz单位向量
- 对于每个运单号，把所有特征的最小值最大值平均数中位数找到

# 福娃是这么做的：
# 把和出发地目的地的经纬度差值找到，再把剩多少时间到下一站(终点站或中转)
# 特征数据的总行数是和相应订单的数据行数加在一起一样多的
# 不过福娃有个比较新颖的办法：设计行程并生成.csv，这样就好计算多段航程的球面距离之和

# 球面距离之和，速率，方向，停船时间比例，低速(<10km/h)运行时间比例
# 和高速(>20km/h)运行时间比例最终被列为我们训练的特征
```

Part5. 训练模型(灵感来自嗷嗷最爱看的b站视频，内容大概是算法优化思路，链接：https://www.bilibili.com/video/BV1mZ4y1j7UJ?t=1089)，在lightGBM的基础上再加上残差迭代训练：

```Python
def build_model(): 
  # 用k-fold的方法去实现效率最高的lightgbm回归(因为比赛排名指标是均方根误差MSE)
  # k = 10 到 25最理想选择，运算既不会太费力，准确度又不低

按路线分解测试集的训练大法
- 把测试集按路线分解以后(其实早在处理训练集的时候就有把测试集分解的打算的)
- 叠加训练(additive training)，每次训练出来result和train这两个表格就多一列nj级残差，nj=12
- 使用nk个种子，nk增长方式按路线内运单号数量的三倍递增，12封顶
- 比较各个迭代残差的方差标准差，从一级残差加到方差加到n_max(n_max < nj)级残差(也就是标准差最小的一级)
- 把迭代残差和测试集预测值加起来，就得最后秒数预测结果
- 取nk个种子的预测结果的q百分位，以避免outliers (如果估早了到达时间ETA就把q调高一些，反之亦然)
- 先把res1(结果)按照订单号字母顺序排列，然后把ETA按测试集B榜每个运单号的列数做个重复，
加上Onboarddate就可以放在test_data并且整理出results
- [ ] 如果要改进算法的话，把加权平均考虑进去就可以了

# 训练模型
res = None
for i in range(len(course_list)): 
    for k in range(nk): 
        for j in range(nj):
            # 不断的迭代nj次训练模型，一共跑nk个不同的种子，
            # 把测试集分解成 ni = len(course_list)份，
            # course_list 是测试集中的路线列表
# 最后再加权平均或者取百分位以后，把分解了的测试集表格组装回去，按一定依据排序使得排序后就是原测试集的行列顺序
```

Part6. 整理输出格式
```Python
test_data.drop(['direction','TRANSPORT_TRACE'],axis=1,inplace=True)
test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
test_data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
test_data['timestamp'] = timestamp.astype(str)
# 整理columns顺序
results = test_data[['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA', 'creatDate']]
results.to_csv('储存路径', index=False)
```

到了复赛阶段，为了应对比赛的复杂形势，我们发明了一个类似于通航行程表和停靠时间表的方法去更好的针对性训练(详见代码xxx)

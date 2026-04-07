import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# 指定目录路径
directory = r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\AgeDB\archive\jpg'

# 初始化一个列表来存储提取的信息
data = []

# 遍历目录下的所有文件
for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        # 假设文件名格式为 'id_name_age_gender'，如 '1_MariaCallas_40_f'
        # 分割文件名（去掉扩展名，如果有的话）
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        
        if len(parts) >= 3:
            # 提取身份（假设是 id + name，如 '1_MariaCallas'）
            name = parts[1]
            # 提取年龄（倒数第二个部分）
            age = parts[-2]
            # 文件名
            file_name = filename
            
            # 将信息添加到列表中
            data.append({
                'filename': file_name,
                'name': name,
                'age': int(age)
            })

# 创建pandas DataFrame
df = pd.DataFrame(data)
print(df.dtypes)

# 打印DataFrame（或你可以保存到CSV/Pickle等）
print(df.head())
ages,names,filenames=df['age'].to_numpy(),df['name'].to_numpy(),df['filename'].to_numpy()
plt.hist(ages.astype(int), bins=range(0, 101, 5), edgecolor='black')
#plt.show()

index_to_name = {i: name for i, name in enumerate(np.unique(names))}
name_to_index = {name: i for i, name in enumerate(np.unique(names))}
indexs = np.array([name_to_index[name] for name in names])
#print(len(index_to_name), len(name_to_index))

gallery_set=filenames[(ages<40) & (indexs<60)]
gallery_indexs=indexs[(ages<40) & (indexs<60)]
probe_set=filenames[(ages>55) & (indexs<60)]
probe_indexs=indexs[(ages>55) & (indexs<60)]
train_set=filenames[indexs>=60]
train_indexs=indexs[indexs>=60]
trainvalid_gallery_set=filenames[(ages<40) & (indexs>=60) & (indexs<70)]
trainvalid_gallery_indexs=indexs[(ages<40) & (indexs>=60) & (indexs<70)]
trainvalid_probe_set=filenames[(ages>55) & (indexs>=60) & (indexs<70)]
trainvalid_probe_indexs=indexs[(ages>55) & (indexs>=60) & (indexs<70)]
#print(len(probe_set), len(gallery_set))
#print(len(train_set))
#print(len(trainvalid_probe_set), len(trainvalid_gallery_set))

#先制作gnd_agedb
gnd_agedb={}
gnd_agedb['imlist']=np.array(gallery_set)
gnd_agedb['imlabel']=gallery_indexs
gnd_agedb['qimlist']=np.array(probe_set)
gnd_agedb['index']=probe_indexs
gnd_agedb['gnd']=[{'ok': [], 'bbx': [], 'junk': [], 'irrel': []} for _ in range(len(probe_set))]
filter_qimlist=np.zeros(len(probe_set), dtype=bool)
for i in range(len(probe_set)):
    rel_list=np.where(gallery_indexs==probe_indexs[i])[0]
    if len(rel_list)==0:
        filter_qimlist[i]=True
    gnd_agedb['gnd'][i]['ok']=rel_list
    gnd_agedb['gnd'][i]['bbx']=[1,2,3,4]
    gnd_agedb['gnd'][i]['junk']=[]
    irrel_list=np.where(gallery_indexs!=probe_indexs[i])[0]
    gnd_agedb['gnd'][i]['irrel']=irrel_list
gnd_agedb['qimlist']=gnd_agedb['qimlist'][~filter_qimlist]
gnd_agedb['index']=gnd_agedb['index'][~filter_qimlist].tolist()
gnd_agedb['gnd']= [gnd_agedb['gnd'][i] for i in range(len(probe_set)) if not filter_qimlist[i]]
#print(gnd_agedb['qimlist'].shape)

#print(len(gnd_agedb['imlist']), len(gnd_agedb['qimlist']))
#print(gnd_agedb['imlist'][gnd_agedb['gnd'][2]['ok']])
#print(gnd_agedb['qimlist'][2])
#print(len(gnd_agedb['gnd']))
path=r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\AgeDB\gnd_agedb.pkl'
with open(path, 'wb') as f:
    pickle.dump(gnd_agedb, f)

#再制作agedb_train
gnd_agedb={}
gnd_agedb['imlist']=np.array(train_set)
gnd_agedb['qimlist']=[]
for i in range(507):
    gnd_agedb['qimlist'].append(str(i+1))
gnd_agedb['qimlist']=np.array(gnd_agedb['qimlist'])
gnd_agedb['gnd']=[{'ok': [], 'bbx': [], 'junk': [], 'irrel': []} for _ in range(507)]
for i in range(507):
    rel_list=np.where(train_indexs==i+60)[0]
    gnd_agedb['gnd'][i]['ok']=rel_list
    gnd_agedb['gnd'][i]['bbx']=[1,2,3,4]
    gnd_agedb['gnd'][i]['junk']=[]
    irrel_list=np.where(train_indexs!=i+60)[0]
    gnd_agedb['gnd'][i]['irrel']=irrel_list
#print(gnd_agedb['imlist'][gnd_agedb['gnd'][2]['ok']])
#print(index_to_name[62])
#print(len(gnd_agedb['imlist']), len(gnd_agedb['qimlist']))
#print(len(gnd_agedb['gnd']))
path=r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\AgeDB\agedb_train.pkl'
with open(path, 'wb') as f:
    pickle.dump(gnd_agedb, f)

#再制作agedb_trainvalid
gnd_agedb={}
gnd_agedb['imlist']=np.array(trainvalid_gallery_set)
gnd_agedb['imlabel']=trainvalid_gallery_indexs
gnd_agedb['qimlist']=np.array(trainvalid_probe_set)
gnd_agedb['index']=trainvalid_probe_indexs
gnd_agedb['gnd']=[{'ok': [], 'bbx': [], 'junk': [], 'irrel': []} for _ in range(len(trainvalid_probe_set))]
filter_qimlist=np.zeros(len(trainvalid_probe_set), dtype=bool)
for i in range(len(trainvalid_probe_set)):
    rel_list=np.where(trainvalid_gallery_indexs==trainvalid_probe_indexs[i])[0]
    if len(rel_list)==0:
        filter_qimlist[i]=True
    gnd_agedb['gnd'][i]['ok']=rel_list
    gnd_agedb['gnd'][i]['bbx']=[1,2,3,4]
    gnd_agedb['gnd'][i]['junk']=[]
    irrel_list=np.where(trainvalid_gallery_indexs!=trainvalid_probe_indexs[i])[0]
    gnd_agedb['gnd'][i]['irrel']=irrel_list
gnd_agedb['qimlist']=gnd_agedb['qimlist'][~filter_qimlist]
gnd_agedb['index']=gnd_agedb['index'][~filter_qimlist].tolist()
gnd_agedb['gnd']= [gnd_agedb['gnd'][i] for i in range(len(trainvalid_probe_set)) if not filter_qimlist[i]]
print(gnd_agedb['qimlist'].shape)

#print(len(gnd_agedb['imlist']), len(gnd_agedb['qimlist']))
#print(gnd_agedb['imlist'][gnd_agedb['gnd'][2]['ok']])
#print(gnd_agedb['qimlist'][2])
#print(len(gnd_agedb['gnd']))
path=r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\AgeDB\agedb_trainvalid.pkl'
with open(path, 'wb') as f:
    pickle.dump(gnd_agedb, f)
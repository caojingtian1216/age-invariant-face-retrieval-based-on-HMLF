import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 指定目录路径
base_dir = r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\imdb-clean-1024\jpg'

# 用于存储结果的列表
data = []

# 正则表达式匹配文件名格式：nmXXXXXXX_rmYYYYYYYY_YYYY-MM-DD_YYYY
# 假设文件名以 nm 开头，后跟 ID，然后 _rm..._birthdate_shootyear
# 示例：nm0000100_rm46373120_1955-1-6_2003
pattern = re.compile(r'(nm\d+)_rm\d+_(\d{4})-\d+-\d+_(\d{4})')

# 遍历目录及其子目录
for root, dirs, files in os.walk(base_dir):
    for file in files:
        # 检查是否是图片文件（假设扩展名是 .jpg, .png 等）
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            match = pattern.match(file)
            if match:
                id_part = match.group(1)  # nm0000100
                birth_year = match.group(2)  # 1955
                shoot_year = match.group(3)  # 2003
                # 存储信息：ID, 出生年份, 拍摄年份, 文件路径
                relative_path = os.path.relpath(root, base_dir)
                filename = os.path.join(relative_path, file)
                filename=filename.replace('\\', '/')
                data.append({
                    'ID': id_part,
                    'age': int(shoot_year) - int(birth_year),
                    'Filename': filename
                })

# 输出结果（可以打印或保存到文件）
df=pd.DataFrame(data)
print(df.head())
ages,names,filenames=df['age'].to_numpy(),df['ID'].to_numpy(),df['Filename'].to_numpy()
plt.hist(ages.astype(int), bins=range(0, 101, 5), edgecolor='black')
#plt.show()
#print(len(np.unique(names)))
#print(len(filenames))
#print(max(ages), min(ages))

index_to_name = {i: name for i, name in enumerate(np.unique(names))}
name_to_index = {name: i for i, name in enumerate(np.unique(names))}
indexs = np.array([name_to_index[name] for name in names])
#print(len(index_to_name), len(name_to_index))

gallery_set=filenames[(ages<=40) & (5000<=indexs) & (indexs<5200)]
gallery_indexs=indexs[(ages<=40) & (5000<=indexs) & (indexs<5200)]
probe_set=filenames[(ages>55) & (5000<=indexs) & (indexs<5200)]
probe_indexs=indexs[(ages>55) & (5000<=indexs) & (indexs<5200)]
train_set=filenames[indexs<5000]
train_indexs=indexs[indexs<5000]
trainvalid_gallery_set=filenames[(ages<40) & (indexs<50)]
trainvalid_gallery_indexs=indexs[(ages<40) & (indexs<50)]
trainvalid_probe_set=filenames[(ages>55) & (indexs<50)]
trainvalid_probe_indexs=indexs[(ages>55) & (indexs<50)]
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
#flag_imlist=np.zeros(len(gallery_set), dtype=bool)
for i in range(len(probe_set)):
    rel_list=np.where(gallery_indexs==probe_indexs[i])[0]
    gnd_agedb['gnd'][i]['ok']=rel_list
    if len(rel_list)==0:
        filter_qimlist[i]=True
#    flag_imlist[rel_list]=True
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
path=r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\imdb-clean-1024\gnd_imdb.pkl'
with open(path, 'wb') as f:
    pickle.dump(gnd_agedb, f)

#再制作agedb_train
gnd_agedb={}
gnd_agedb['imlist']=np.array(train_set)
#gnd_agedb['imlabel']=train_indexs
gnd_agedb['qimlist']=[]
for i in range(5000):
    gnd_agedb['qimlist'].append(str(i+1))
gnd_agedb['qimlist']=np.array(gnd_agedb['qimlist'])
gnd_agedb['gnd']=[{'ok': [], 'bbx': [], 'junk': [], 'irrel': []} for _ in range(5000)]
for i in range(5000):
    rel_list=np.where(train_indexs==i)[0]
    gnd_agedb['gnd'][i]['ok']=rel_list
    gnd_agedb['gnd'][i]['bbx']=[1,2,3,4]
    gnd_agedb['gnd'][i]['junk']=[]
    irrel_list=np.where(train_indexs!=i)[0]
    gnd_agedb['gnd'][i]['irrel']=irrel_list
#print(gnd_agedb['imlist'][gnd_agedb['gnd'][2]['ok']])
#print(index_to_name[2])
#print(len(gnd_agedb['imlist']), len(gnd_agedb['qimlist']))
#print(len(gnd_agedb['gnd']))
path=r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\imdb-clean-1024\imdb_train.pkl'
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
#print(gnd_agedb['qimlist'].shape)

#print(len(gnd_agedb['imlist']), len(gnd_agedb['qimlist']))
#print(gnd_agedb['imlist'][gnd_agedb['gnd'][2]['ok']])
#print(gnd_agedb['qimlist'][2])
#print(len(gnd_agedb['gnd']))
path=r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\imdb-clean-1024\imdb_trainvalid.pkl'
with open(path, 'wb') as f:
    pickle.dump(gnd_agedb, f)
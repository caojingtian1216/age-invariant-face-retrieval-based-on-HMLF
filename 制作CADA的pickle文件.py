from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from collections import Counter
import random

# 打开 .mat 文件
mat_file=loadmat(r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\CACD\data\celebrity2000_meta.mat')
print(mat_file.keys())
print(type(mat_file['celebrityData']))
print(mat_file['celebrityData'][0,0][0][52])
print(mat_file['celebrityImageData'].shape)
#文件名称
print(mat_file['celebrityImageData'][0,0][7].shape)
print(type(mat_file['celebrityImageData'][0,0][7]))
file_names=mat_file['celebrityImageData'][0,0][7].reshape(-1)
#print(file_names[10:50])
print(type(file_names[0][0]))

#条件筛选数据集 2010-2012
print((mat_file['celebrityImageData'][0,0][2].reshape(-1)>=2010) & (mat_file['celebrityImageData'][0,0][2].reshape(-1)<=2012))
print(mat_file['celebrityImageData'][0,0][7].reshape(-1)[(mat_file['celebrityImageData'][0,0][2].reshape(-1)>=2010) & (mat_file['celebrityImageData'][0,0][2].reshape(-1)<=2012)].shape)

print("***")
rank=mat_file['celebrityData'][0,0][3].reshape(-1)
print(rank[:5])
print(rank[20:50])
sfjd
'''
plt.figure(figsize=(10, 5))
plt.hist(rank, bins=50, color='blue', alpha=0.7)
plt.title('Rank Distribution')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
'''
id=mat_file['celebrityData'][0,0][1].reshape(-1)
print(id[:5])
#print(np.where((rank>=3) & (rank<=5))[0])  # 获取满足条件的索引
#print(np.where((rank>=3) & (rank<=5))[0].shape)
#print(len(np.where((rank>=3) & (rank<=5))[0]))
#print(len(np.where((rank>5) & (rank<=20))[0]))
print(len(np.where((rank>=3) & (rank<=5))[0]))
print("***")

# 创建CADA的pickle文件
print(min(mat_file['celebrityImageData'][0,0][1].reshape(-1)))
print(max(mat_file['celebrityImageData'][0,0][1].reshape(-1)))
gnd_cada={}
gnd_cada['gnd']=[]
rank_list=np.where(rank>=21)[0] + 1
print(len(rank_list))
print(mat_file['celebrityData'][0,0][0].reshape(-1)[70])

condition3= (mat_file['celebrityImageData'][0,0][2].reshape(-1)>=2004) & (mat_file['celebrityImageData'][0,0][2].reshape(-1)<=2006) & (np.isin(mat_file['celebrityImageData'][0,0][1].reshape(-1), rank_list))
condition4= (mat_file['celebrityImageData'][0,0][2].reshape(-1)==2013) & (np.isin(mat_file['celebrityImageData'][0,0][1].reshape(-1), rank_list))
#condition_rank=np.isin(mat_file['celebrityImageData'][0,0][1].reshape(-1), rank_list)
# 处理imlist
im_list=mat_file['celebrityImageData'][0,0][7].reshape(-1)[condition3]
im_list_id=mat_file['celebrityImageData'][0,0][1].reshape(-1)[condition3]
#print(type(im_list_id[0]))
gnd_cada['imlist']=[]
for i in range(len(im_list)):
    gnd_cada['imlist'].append(str(im_list[i][0]))
print("imlist的长度为：",len(gnd_cada['imlist']))

djks
# 处理qimlist
qim_list=mat_file['celebrityImageData'][0,0][7].reshape(-1)[condition4]
qim_list_id=mat_file['celebrityImageData'][0,0][1].reshape(-1)[condition4]
gnd_cada['qimlist']=[]
for i in range(len(qim_list)):
    gnd_cada['qimlist'].append(str(qim_list[i][0]))
print("qimlist的长度为：",len(gnd_cada['qimlist']))
#print("id的最大值为：",max(mat_file['celebrityImageData'][0,0][1].reshape(-1)))
#print("id的最小值为：",min(mat_file['celebrityImageData'][0,0][1].reshape(-1)))

#qimlist_id=mat_file['celebrityImageData'][0,0][1].reshape(-1)[condition4]
start_time = time.time()
exist=np.zeros(2000)
gnd_cada['gnd']=[{'ok': [], 'bbx': [], 'junk': [], 'irrel': []} for _ in range(2000)]
for i in range(len(gnd_cada['qimlist'])):
    cur_id=qim_list_id[i]
    rel_list=np.where(im_list_id==cur_id)[0]
    gnd_cada['gnd'][i]['ok']=rel_list
    gnd_cada['gnd'][i]['bbx']=[1,2,3,4]
    gnd_cada['gnd'][i]['junk']=[]
    irrel_list=np.where(im_list_id!=cur_id)[0]
    if exist[cur_id-1]==0:  #注意：下标有减一
#        gnd_cada['gnd'][cur_id-1]['ok']=rel_list
        gnd_cada['gnd'][cur_id-1]['irrel']=irrel_list
        exist[cur_id-1]=1

end_time = time.time()
print(f"运行时间: {end_time - start_time:.6f} 秒")
#print(qimlist_id[:5])
print(gnd_cada['gnd'][0]['ok'])
print(gnd_cada['gnd'][2]['irrel'])
print(len(gnd_cada['gnd'][2]['irrel'])+len(gnd_cada['gnd'][0]['ok']))
#print("总共相关的个数为：",sum(len(i['ok']) for i in gnd_cada['gnd'] if len(i['ok'])>0))
print(min(len(i['ok']) for i in gnd_cada['gnd'] if len(i['ok'])>0))

gnd_cada['index']=qim_list_id.tolist()
# 保存为pickle文件
for i in range(len(gnd_cada['imlist'])):
    file_name=gnd_cada['imlist'][i]
    # 检查文件名中是否含有单引号
    if "'" in file_name:
        new_file_name=file_name.replace("'","")
        gnd_cada['imlist'][i]=new_file_name

for i in range(len(gnd_cada['qimlist'])):
    file_name=gnd_cada['qimlist'][i]
    # 检查文件名中是否含有单引号
    if "'" in file_name:
        new_file_name=file_name.replace("'","")
        gnd_cada['qimlist'][i]=new_file_name
         
#with open(r'C:\Users\cao\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\CACD\data\gnd_cada1.pkl', 'wb') as f:
#    pickle.dump(gnd_cada, f)

#counter1=Counter(qimlist_id)
#print("qimlist_id中每个id出现的次数：", counter1)
#counter2=Counter(im_list_id)
#print("im_list_id中每个id出现的次数：", counter2)

select_imlist=np.array(random.sample(range(3727), 300))
train_imlist = [gnd_cada['imlist'][i] for i in range(3727) if i not in select_imlist]
train_imlist_id = [im_list_id[i] for i in range(3727) if i not in select_imlist]
query_imlist = [gnd_cada['imlist'][i] for i in select_imlist]
query_imlist_id = [im_list_id[i] for i in select_imlist]
print("train_imlist的长度为：", len(train_imlist))
print("query_imlist的长度为：", len(query_imlist))

exist=np.zeros(2000)
gnd_cada['imlist']=train_imlist
gnd_cada['qimlist']=query_imlist
gnd_cada['index']=query_imlist_id
gnd_cada['gnd']=[{'ok': [], 'bbx': [], 'junk': [], 'irrel': []} for _ in range(2000)]
for i in range(300):
    cur_id=query_imlist_id[i]
    rel_list=np.where(train_imlist_id==cur_id)[0]
    gnd_cada['gnd'][i]['ok']=rel_list
    gnd_cada['gnd'][i]['bbx']=[1,2,3,4]
    gnd_cada['gnd'][i]['junk']=[]
    irrel_list=np.where(train_imlist_id!=cur_id)[0]
    if exist[cur_id-1]==0:  #注意：下标有减一
        gnd_cada['gnd'][cur_id-1]['irrel']=irrel_list
        exist[cur_id-1]=1

print(len(gnd_cada['gnd'][0]['ok']))
print(len(gnd_cada['gnd'][gnd_cada['index'][0]-1]['irrel']))
print(len(gnd_cada['gnd'][0]['ok']) + len(gnd_cada['gnd'][gnd_cada['index'][0]-1]['irrel']))
# 保存为pickle文件
#with open(r'C:\Users\cao\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\CACD\data\gnd_train_cada3.pkl', 'wb') as f:
#    pickle.dump(gnd_cada, f)

counter1=Counter(query_imlist_id)
print("query_imlist_id中每个id出现的次数：", counter1)
counter2=Counter(train_imlist_id)
print("train_imlist_id中每个id出现的次数：", counter2)



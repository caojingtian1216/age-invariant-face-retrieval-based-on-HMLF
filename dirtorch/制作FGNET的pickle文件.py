import os
import re
import pandas as pd
import numpy as np
import pickle
import random

FGNET_ROOT = r"C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\FGNET\jpg"
FGNET_ROOT=r".\dirtorch\data\datasets\FGNET\jpg"

pattern = re.compile(r"(?P<id>\d{3})A(?P<age>\d{2})", re.IGNORECASE)
records = []

for filename in os.listdir(FGNET_ROOT):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        continue

    match = pattern.match(name)
    if not match:
        continue

    identity = int(match.group("id"))
    age = int(match.group("age"))
    records.append({
        "identity": identity,
        "age": age,
        "filename": filename,
        "filepath": os.path.join(FGNET_ROOT, filename)
    })

df = pd.DataFrame(records)
print(df.head())
print(f"共解析 {len(df)} 张图像。")
imlist=df['filename'].tolist()
label=df['identity'].tolist()

for img_id in range(1002):
    if img_id>5:
        break
    #先制作gnd_fgnet
    imlist=df['filename'].tolist()
    label=df['identity'].tolist()
    train_imlist=np.array(imlist[:img_id]+imlist[img_id+1:])
    test_imlist=np.array([imlist[img_id]])
    train_label=np.array(label[:img_id]+label[img_id+1:])
    test_label=np.array([label[img_id]])

    gnd_fgnet={}
    gnd_fgnet['imlist']=np.array(train_imlist)
    gnd_fgnet['imlabel']=train_label
    gnd_fgnet['qimlist']=np.array(test_imlist)
    gnd_fgnet['index']=test_label.tolist()
    gnd_fgnet['gnd']=[{'ok': [], 'bbx': [], 'junk': [], 'irrel': []} for _ in range(1)]
    for i in range(1):
        rel_list=np.where(train_label==test_label[i])[0]
        gnd_fgnet['gnd'][i]['ok']=rel_list
        gnd_fgnet['gnd'][i]['bbx']=[1,2,3,4]
        gnd_fgnet['gnd'][i]['junk']=[]
        irrel_list=np.where(train_label!=test_label[i])[0]
        gnd_fgnet['gnd'][i]['irrel']=irrel_list

    #path=r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\FGNET\gnd_fgnet.pkl'
    #with open(path, 'wb') as f:
    #    pickle.dump(gnd_fgnet, f)
    
    #再制作fgnet_train
    gnd_fgnet['imlist']=train_imlist[train_label!=label[img_id]]
    new_label=train_label[train_label!=label[img_id]]
    gnd_fgnet['qimlist']=[]
    for i in range(82):
        gnd_fgnet['qimlist'].append(str(i+1))
    gnd_fgnet['qimlist']=np.array(gnd_fgnet['qimlist'])
    gnd_fgnet['gnd']=[{'ok': [], 'bbx': [], 'junk': [], 'irrel': []} for _ in range(82)]
    del gnd_fgnet['index']
    for i in range(82):
        rel_list=np.where(new_label==i+1)[0]
        gnd_fgnet['gnd'][i]['ok']=rel_list
        gnd_fgnet['gnd'][i]['bbx']=[1,2,3,4]
        gnd_fgnet['gnd'][i]['junk']=[]
        irrel_list=np.where(new_label!=i+1)[0]
        gnd_fgnet['gnd'][i]['irrel']=irrel_list
#    print(gnd_fgnet['imlist'][gnd_fgnet['gnd'][3]['ok']])
#    print("number of valid identities:",cnt)
    #path=r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\FGNET\fgnet_train.pkl'
    #with open(path, 'wb') as f:
    #    pickle.dump(gnd_fgnet, f)
    
    #再制作fgnet_trainvalid
    numbers = random.sample(range(0, 1001), 50)
    test_imlist=train_imlist[numbers]
    test_label=train_label[numbers]
    gnd_fgnet['imlist']=np.array(train_imlist)
    gnd_fgnet['qimlist']=test_imlist
    gnd_fgnet['imlabel']=train_label
    gnd_fgnet['index']=test_label.tolist()
    gnd_fgnet['gnd']=[{'ok': [], 'bbx': [], 'junk': [], 'irrel': []} for _ in range(50)]
    for i in range(50):
        rel_list=np.where(train_label==test_label[i])[0]
        gnd_fgnet['gnd'][i]['ok']=rel_list
        gnd_fgnet['gnd'][i]['bbx']=[1,2,3,4]
        gnd_fgnet['gnd'][i]['junk']=[]
        irrel_list=np.where(train_label!=test_label[i])[0]
        gnd_fgnet['gnd'][i]['irrel']=irrel_list
    print(gnd_fgnet['imlist'][gnd_fgnet['gnd'][3]['ok']])
    dfksj
    #path=r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\FGNET\fgnet_trainvalid.pkl'
    #with open(path, 'wb') as f:
    #    pickle.dump(gnd_fgnet, f)

    
        

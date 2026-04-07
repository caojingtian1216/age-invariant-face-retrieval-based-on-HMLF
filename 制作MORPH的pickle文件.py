import os
import pickle
import numpy as np
import re
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def list_image_filenames(dir_path):
	exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}
	if not os.path.isdir(dir_path):
		raise FileNotFoundError(f"目录不存在: {dir_path}")
	files = []
	for name in os.listdir(dir_path):
		p = os.path.join(dir_path, name)
		if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
			files.append(name)
	files.sort()
	return files


target_dir = r"C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\morph\data\jpg"
names = list_image_filenames(target_dir)
print(f"图片数量: {len(names)}")
for n in names[:5]:
    print(n)

def parse_morph_filename_regex(filename):
    """
    使用正则表达式解析MORPH文件名
    """
    # 匹配模式：6位数字_数字+字母+2位数字
    pattern = r'^(\d{6})_(\d+)([MF])(\d{2}).JPG$'
    match = re.match(pattern, filename)
    
    if match:
        subject_id = match.group(1)
        sequence_num = match.group(2)  # 序列号（您可能不需要）
        gender = match.group(3)
        age = int(match.group(4))
        return subject_id, gender, age
    else:
        print(f"警告: 文件名格式不匹配: {filename}")
        return None, None, None

# 解析所有文件名
parsed_data = []
for filename in names:
    result = parse_morph_filename_regex(filename)
    if result[0] is not None:  # 只添加成功解析的文件
        subject_id, gender, age = result
        parsed_data.append({
            'Filename': filename,
            'Subject_ID': subject_id,
            'Gender': gender,
            'Age': age
        })

df_regex = pd.DataFrame(parsed_data)
print("使用正则表达式解析的结果:")
print(df_regex.head())
print(df_regex['Subject_ID'].nunique())
print(df_regex['Subject_ID'].dtype)

arr = df_regex['Subject_ID'].unique()  # 假设有12938个元素

group_1 = np.random.choice(arr, 10000, replace=False).tolist()
remaining = [x for x in arr if x not in group_1]
group_2 = np.random.choice(remaining, 600, replace=False).tolist()
group_3 = np.random.choice(group_2, 100, replace=False).tolist()


def get_extreme_age_images(group):
    """为每个受试者组找到年龄最小和最大的照片"""
    min_age = group['Age'].min()
    max_age = group['Age'].max()
    
    # 选取年龄最小的照片（如果有多个相同年龄，取第一个）
    min_age_row = group[group['Age'] == min_age].iloc[0]
    # 选取年龄最大的照片（如果有多个相同年龄，取最后一个）
    max_age_row = group[group['Age'] == max_age].iloc[-1]
    
    return pd.Series({
        'youngest_filename': min_age_row['Filename'],
        'youngest_age': min_age_row['Age'],
        'oldest_filename': max_age_row['Filename'],
        'oldest_age': max_age_row['Age'],
        'total_images': len(group),
        'age_range': max_age - min_age
    })

# 按受试者分组并应用函数
subject_summary = df_regex.groupby('Subject_ID').apply(get_extreme_age_images).reset_index()

print(f"  最小年龄跨度: {subject_summary['age_range'].min()} 年")
print(f"  最大年龄跨度: {subject_summary['age_range'].max()} 年")
print(f"  平均年龄跨度: {subject_summary['age_range'].mean():.2f} 年")

#selected_subjects = subject_summary.sample(n=10000, random_state=RANDOM_SEED)
selected_subjects = subject_summary[subject_summary['Subject_ID'].isin(group_1)]
print(f"\n成功随机选择了 {len(selected_subjects)} 个受试者")


index_to_ID={i: id for i, id in enumerate(sorted(selected_subjects['Subject_ID'].unique()), start=1)}
ID_to_index={id: i for i, id in enumerate(sorted(selected_subjects['Subject_ID'].unique()), start=1)}
print(index_to_ID[1])

# 分别提取 youngest 和 oldest 的文件名列表
youngest_filenames = selected_subjects[['Subject_ID', 'youngest_filename', 'youngest_age']].rename(
    columns={'youngest_filename': 'Filename', 'youngest_age': 'Age'})

oldest_filenames = selected_subjects[['Subject_ID', 'oldest_filename', 'oldest_age']].rename(
    columns={'oldest_filename': 'Filename', 'oldest_age': 'Age'})

gnd_morph = {}
#youngest_filenames和oldest_filenames各10000张图片
im_list=[]
im_list_index=[]
qim_list=[]
qim_list_index=[]
gnd_morph['imlist']=[]
for i in range(len(youngest_filenames)):
    im_list.append(youngest_filenames.iloc[i]['Filename'])
    im_list_index.append(ID_to_index[youngest_filenames.iloc[i]['Subject_ID']])
    gnd_morph['imlist'].append(youngest_filenames.iloc[i]['Filename'])
im_list=np.array(im_list)
im_list_index=np.array(im_list_index)
gnd_morph['imlist']=np.array(gnd_morph['imlist'])

gnd_morph['qimlist']=[]
for i in range(len(oldest_filenames)):
    qim_list.append(oldest_filenames.iloc[i]['Filename'])
    qim_list_index.append(ID_to_index[oldest_filenames.iloc[i]['Subject_ID']])
    gnd_morph['qimlist'].append(oldest_filenames.iloc[i]['Filename'])
qim_list=np.array(qim_list)
qim_list_index=np.array(qim_list_index)
gnd_morph['qimlist']=np.array(gnd_morph['qimlist'])


exist=np.zeros(10000)
gnd_morph['gnd']=[{'ok': [], 'bbx': [], 'junk': [], 'irrel': []} for _ in range(10000)]
for i in range(len(gnd_morph['qimlist'])):
    cur_id=qim_list_index[i]
    rel_list=np.where(im_list_index==cur_id)[0]
    gnd_morph['gnd'][i]['ok']=rel_list
    gnd_morph['gnd'][i]['bbx']=[1,2,3,4]
    gnd_morph['gnd'][i]['junk']=[]
    irrel_list=np.where(im_list_index!=cur_id)[0]
    if exist[cur_id-1]==0:  #注意：下标有减一
#        gnd_morph['gnd'][cur_id-1]['ok']=rel_list
        gnd_morph['gnd'][cur_id-1]['irrel']=irrel_list
        exist[cur_id-1]=1


gnd_morph['index']=qim_list_index.tolist()
# 保存为pickle文件
#path=r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\morph\data\gnd_morph.pkl'
#with open(path, 'wb') as f:
#    pickle.dump(gnd_morph, f)

selected_subjects=df_regex[df_regex['Subject_ID'].isin(group_2)]
print(f"\n成功随机选择了 {len(selected_subjects)} 张图片用于训练")

print(selected_subjects.head())

#总共600人
index_to_ID={i: id for i, id in enumerate(sorted(selected_subjects['Subject_ID'].unique()), start=1)}
ID_to_index={id: i for i, id in enumerate(sorted(selected_subjects['Subject_ID'].unique()), start=1)}
print(index_to_ID[1])
n_people=600
gnd_morph_train = {}
imlist=[]
qimlist=[]
gnd_morph['imlist']=[]
for i in range(len(selected_subjects)):
    imlist.append(selected_subjects.iloc[i]['Filename'])
    gnd_morph['imlist'].append(selected_subjects.iloc[i]['Filename'])
gnd_morph['qimlist']=[]
for i in range(n_people):
    qimlist.append(str(i+1))
    gnd_morph['qimlist'].append(str(i+1))
gnd_morph['imlist']=np.array(gnd_morph['imlist'])
gnd_morph['qimlist']=np.array(gnd_morph['qimlist'])
gnd_morph['gnd']=[{'ok': [], 'bbx': [], 'junk': [], 'irrel': []} for _ in range(600)]

selected_subjects = selected_subjects.reset_index(drop=True)
for i in range(600):
    rel_list=selected_subjects[selected_subjects['Subject_ID']==index_to_ID[i+1]].index.tolist()
#    if index_to_ID[i+1]=='025119':
#        print(selected_subjects[selected_subjects['Subject_ID']==index_to_ID[i+1]])
#        print('***')
#        print(rel_list)
#        print(selected_subjects.iloc[rel_list])
#        print(i)
#        break
    gnd_morph['gnd'][i]['ok']=rel_list
    gnd_morph['gnd'][i]['bbx']=[1,2,3,4]
    gnd_morph['gnd'][i]['junk']=[]
    irrel_list=selected_subjects[selected_subjects['Subject_ID']!=index_to_ID[i+1]].index.tolist()
    gnd_morph['gnd'][i]['irrel']=irrel_list

#print(gnd_morph['imlist'][rel_list])


# 保存为pickle文件
#path=r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\morph\data\morph_train.pkl'
#with open(path, 'wb') as f:
#    pickle.dump(gnd_morph, f)

selected_subjects = subject_summary[subject_summary['Subject_ID'].isin(group_3)]
print(f"\n成功随机选择了 {len(selected_subjects)} 个受试者")


index_to_ID={i: id for i, id in enumerate(sorted(selected_subjects['Subject_ID'].unique()), start=1)}
ID_to_index={id: i for i, id in enumerate(sorted(selected_subjects['Subject_ID'].unique()), start=1)}
print(index_to_ID[1])

# 分别提取 youngest 和 oldest 的文件名列表
youngest_filenames = selected_subjects[['Subject_ID', 'youngest_filename', 'youngest_age']].rename(
    columns={'youngest_filename': 'Filename', 'youngest_age': 'Age'})

oldest_filenames = selected_subjects[['Subject_ID', 'oldest_filename', 'oldest_age']].rename(
    columns={'oldest_filename': 'Filename', 'oldest_age': 'Age'})

gnd_morph = {}
#youngest_filenames和oldest_filenames各100张图片
im_list=[]
im_list_index=[]
qim_list=[]
qim_list_index=[]
gnd_morph['imlist']=[]
for i in range(len(youngest_filenames)):
    im_list.append(youngest_filenames.iloc[i]['Filename'])
    im_list_index.append(ID_to_index[youngest_filenames.iloc[i]['Subject_ID']])
    gnd_morph['imlist'].append(youngest_filenames.iloc[i]['Filename'])
im_list=np.array(im_list)
im_list_index=np.array(im_list_index)
gnd_morph['imlist']=np.array(gnd_morph['imlist'])

gnd_morph['qimlist']=[]
for i in range(len(oldest_filenames)):
    qim_list.append(oldest_filenames.iloc[i]['Filename'])
    qim_list_index.append(ID_to_index[oldest_filenames.iloc[i]['Subject_ID']])
    gnd_morph['qimlist'].append(oldest_filenames.iloc[i]['Filename'])
qim_list=np.array(qim_list)
qim_list_index=np.array(qim_list_index)
gnd_morph['qimlist']=np.array(gnd_morph['qimlist'])
gnd_morph['imlabel']=im_list_index.tolist()


exist=np.zeros(100)
gnd_morph['gnd']=[{'ok': [], 'bbx': [], 'junk': [], 'irrel': []} for _ in range(10000)]
for i in range(len(gnd_morph['qimlist'])):
    cur_id=qim_list_index[i]
    rel_list=np.where(im_list_index==cur_id)[0]
    gnd_morph['gnd'][i]['ok']=rel_list
    gnd_morph['gnd'][i]['bbx']=[1,2,3,4]
    gnd_morph['gnd'][i]['junk']=[]
    irrel_list=np.where(im_list_index!=cur_id)[0]
    if i==5:
        print(rel_list)
    if exist[cur_id-1]==0:  #注意：下标有减一
#        gnd_morph['gnd'][cur_id-1]['ok']=rel_list
        gnd_morph['gnd'][cur_id-1]['irrel']=irrel_list
        exist[cur_id-1]=1


gnd_morph['index']=qim_list_index.tolist()
# 保存为pickle文件
#path=r'C:\Users\surface\Desktop\deep-image-retrieval-master\dirtorch\data\datasets\morph\data\morph_trainvalid.pkl'
#with open(path, 'wb') as f:
#    pickle.dump(gnd_morph, f)



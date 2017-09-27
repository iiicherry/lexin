import pandas as pd

ud = pd.read_csv(r'E:\ud_mdl.csv', header=0, index_col=0)
dep = pd.read_csv(r'E:\dep_mdl.csv', header=0, index_col=0, usecols=[0,1])
#join data according to user ID
train = pd.concat([ud, dep], axis=1)

#drop unecessary data--------------------------------------
train = train.drop(['fschoolarea_name_md5','fcal_graduation','sch_fcompany_name'],1)
#register time & pocket auth time---------------------------
train['fregister_time'] = train['fregister_time'].str.split(':').str.get(0)
train['fregister_time'] = pd.to_datetime(train['fregister_time'],format='%d%b%y')
train['rtime'] = pd.datetime(2016,10,1)-train['fregister_time']
train = train.drop('fregister_time',1) 
#time_d.days|time_d_float = time_d.total_seconds()
train['fpocket_auth_time'] = train['fpocket_auth_time'].str.split(':').str.get(0)
train['fpocket_auth_time'] = pd.to_datetime(train['fpocket_auth_time'],format='%d%b%y')
train['atime'] = pd.datetime(2016,10,1)-train['fpocket_auth_time']
train = train.drop('fpocket_auth_time',1) 
#xgb doesn't accept timedelta
train['rtime'] = train['rtime']/ pd.Timedelta(days=1)
train['atime'] = train['atime']/ pd.Timedelta(days=1)

#Geogpraphic info: hometown---------------------------------
#fill in nulls
train['fdomicile_provice'].fillna(train['sch_fprovince_name'],inplace=True)
train['fdomicile_city'].fillna(train['sch_fcity_name'],inplace=True)
train['fdomicile_area'].fillna(train['sch_fregion_name'],inplace=True)

train.dropna(axis=0,inplace=True)
#in high default rate province?
bad_prov = train[['fdomicile_provice', 'dep']].groupby('fdomicile_provice').mean().\
sort_values(by='dep')[-5:].index.tolist()
train['dprov'] = train['fdomicile_provice'].isin(bad_prov).astype(int)
#城市区间
percent = train[['fdomicile_city', 'dep']].groupby(['fdomicile_city'],as_index=True)\
.mean().sort_values(by='dep')
percent['dep2']= pd.cut(percent.dep, 10, labels=[0,1,2,3,4,5,6,7,8,9])
citydict = percent['dep2'].to_dict()
train['dcity'] = train['fdomicile_city'].map(citydict) #note city null values
#dcity_dummies = pd.get_dummies(train['dcity'])
#dcity_dummies.columns = ['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9']
#train = train.join(dcity_dummies)
#train = train.drop('dcity',1)  

#city rank
def cityrank(x):
    if x in ['北京市','上海市','广州市','深圳市']:
        return 0
    else:
        if x in ['成都市','杭州市','武汉市','重庆市','南京市','天津市','苏州市','西安市','长沙市',\
        '沈阳市','青岛市','郑州市','大连市','东莞市','宁波市']:
            return 1
        else:
            if x in ['厦门市','福州市','无锡市','合肥市','昆明市','哈尔滨市','济南市','佛山市','长春市',\
            '温州市','石家庄市','南宁市','常州市','泉州市','南昌市','贵阳市','太原市','烟台市','嘉兴市',\
            '南通市','金华市','珠海市','惠州市','徐州市','海口市','乌鲁木齐市','绍兴市','中山市','台州市','兰州市']:
                return 2
            else:
                return 3
train['dcityrank'] = train['fdomicile_city'].apply(cityrank)
drank_dummies = pd.get_dummies(train['dcityrank'])
drank_dummies.columns = ['drank0','drank1','drank2','drank3']
train = train.join(drank_dummies)

#town or city?
def istown(x):
    if u'县' in x:
        return 1
    else:
        return 0
train['dtown'] = train['fdomicile_area'].apply(istown)
#note null values：missing values would get interpreted as float and would raise the TypeError on iterating
#map applymap
train = train.drop('fdomicile_area',1)

#Geogpraphic info: school---------------------------------
#in high default rate province?
bad_sprov = train[['sch_fprovince_name', 'dep']].groupby('sch_fprovince_name')\
.mean().sort_values(by='dep')[-5:].index.tolist()
train['sprov'] = train['sch_fprovince_name'].isin(bad_sprov).astype(int)
#城市区间
percent = train[['sch_fprovince_name', 'dep']].groupby('sch_fprovince_name')\
.mean().sort_values(by='dep')
percent['dep2']= pd.cut(percent.dep, 10, labels=[0,1,2,3,4,5,6,7,8,9])
scitydict = percent['dep2'].to_dict()
train['scity'] = train['sch_fprovince_name'].map(scitydict) #note city null values
#scity_dummies = pd.get_dummies(train['scity'])
#scity_dummies.columns = ['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9']
#train = train.join(scity_dummies)
#train = train.drop('scity',1) 

#school city rank
train['scityrank'] = train['sch_fprovince_name'].apply(cityrank)
srank_dummies = pd.get_dummies(train['scityrank'])
srank_dummies.columns = ['srank0','srank1','srank2']
train = train.join(srank_dummies)

#school, town or city?
train['stown'] = train['sch_fregion_name'].apply(istown)
train = train.drop('sch_fregion_name',1)
#上学是否跨省
train['provdiff'] = (train['fdomicile_provice'] == train['sch_fprovince_name']).astype(int)
train = train.drop(['fdomicile_provice','sch_fprovince_name'],1)
#上学是否降级
train['rankdiff'] = train['scityrank']-train['dcityrank']
train = train.drop(['fdomicile_city','sch_fcity_name','dcityrank','scityrank'],1)

#fsex
train['fsex'].replace(2, 1, inplace=True)
#fis_entrance_exam
train['fis_entrance_exam'].replace(-1, 0, inplace=True)
#fcollege_level
train['fcollege_level'] = pd.Categorical(train['fcollege_level']).codes
college_dummies = pd.get_dummies(train['fcollege_level'])
college_dummies.columns = ['college0','college1','college2','college3','college4','college5','college6']
train = train.join(college_dummies)
train = train.drop('fcollege_level',1)
#fauth_source_type
train['fauth_source_type'] = pd.Categorical(train['fauth_source_type']).codes
auth_dummies = pd.get_dummies(train['fauth_source_type'])
auth_dummies.columns = ['auth0','auth1','auth2','auth3','auth4','auth5','auth6','auth7','auth8','auth9']
train = train.join(auth_dummies)
train = train.drop('fauth_source_type',1)

#wrtie to csv file
cols = [x for x in train.columns if x not in ['dep']]
train.to_csv(r'E:\udtrain.csv', columns=predictors)

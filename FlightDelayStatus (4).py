#!/usr/bin/env python
# coding: utf-8

# <font color='brown'></font> 
# 

# # <font color='Brown'>Flight Delay Prediction<font> 
#     
#     

# Flights are said to be delayed when they arrive later than the scheduled arrival time. This delay is most likely to be  influenced by the environmental conditions. Flight delay is a frustration for passengers and also incurs high financial losses to the airlines and the respected countries. A structured prediction system is a tool that can help aviation authorities effectively predict flight delays. This project aims to build a machine learning engine to effectively predict the arrival delay of a flight in minutes after departure based on real-time flight and weather data.

# The ability to predict a delay in flight can be helpful for all, including airlines and passengers. This study explores the method of predicting flight delay by classifying a specific flight as either delay(1) or no delay(2). From the initial review, the flight delay dataset is skewed. It is expected since most airlines usually have more non-delayed flights than delayed ones. Hence, this study compares different methods to deal with an imbalanced dataset by training a flight delay prediction model with the hourly weather dataset of 2004 and then testing it with the hourly data of 2005 along with the flight test data.

# In[148]:


# Import necessary libraries
import pandas as pd
import numpy as np


# In[149]:


pd.set_option('display.max_columns',None)


# In[150]:


#Flightdata
train=pd.read_csv("train.csv")


# In[151]:


train.head()


# In[152]:


train[['ActualDate','ActualTime']]=train['ActualArrivalTimeStamp'].str.split(" ",expand=True)


# In[153]:


train[['A','B']] = train['ActualTime'].str.split(':',expand=True)


# In[154]:


train['Actual']=train['A']+train['B']


# In[155]:


train.drop(['A','B'],axis=1,inplace=True)


# In[156]:


train['Actual']=train['Actual'].astype(int)


# In[157]:


train['ScheduledArrTime']=train['ScheduledArrTime'].astype(int)


# In[158]:


train['FlightDelayStatus']=train['Actual']-train['ScheduledArrTime']


# In[159]:


#Creating the target variable

def FlightDelaystatus(x):
    if x>=15:
        return 1
    else:
        return 2

train['FlightDelaystatus']=train['FlightDelayStatus'].apply(FlightDelaystatus)
        


# In[160]:


train.head()


# In[161]:


train.FlightDelaystatus.value_counts()


# In[162]:


train.shape


# In[163]:


all_station=pd.read_csv("AllStationsData_PHD.csv")


# In[164]:


all_station.head(2)


# In[165]:


all_station.shape


# In[166]:


#left join based on destination as we are calculating delay based on the destination
all_train=pd.merge(train,all_station,left_on="Destination",right_on="AirportID",how="left")


# In[167]:


all_train.head()


# In[168]:


all_train.shape


# # Hourly Data of 2004

# In[169]:


jan=pd.read_csv("200401janhourly.txt")
mar=pd.read_csv("200403hourly.txt")
may=pd.read_csv("200405hourly.txt")
jul=pd.read_csv("200407hourly.txt")
sep=pd.read_csv("200409hourly.txt")
nov=pd.read_csv("200411hourly.txt")


# In[170]:


hour=[jan,mar,may,jul,sep,nov]
hourly_data=pd.concat(hour)


# In[171]:


hourly_data.shape


# In[172]:


hourly_data.isnull().sum()


# In[173]:


hourly_data.dropna(axis=0,inplace=True)


# In[174]:


all_train.isnull().sum()


# In[175]:


hourly_data.head(2)


# In[176]:


hourly_data['YearMonthDay']=hourly_data.YearMonthDay.astype(str)
hourly_data['Year']=hourly_data.YearMonthDay.str[0:4]
hourly_data['Month']=hourly_data.YearMonthDay.str[4:6]
hourly_data['Day']=hourly_data.YearMonthDay.str[6:8]


# In[177]:


#zfill fills zeroes before an integer
import datetime 
hourly_data['Date']=pd.to_datetime(hourly_data[["Year", "Month", "Day"]])
hourly_data['Time'] = (pd.to_datetime(hourly_data['Time'].astype(str).str.zfill(4), format='%H%M').dt.strftime('%H%M'))


# https://stackoverflow.com/questions/47097447/convert-string-to-hhmm-time-in-python

# In[178]:


hourly_data['Time']=pd.to_datetime(hourly_data['Time'], format='%H%M')


# In[179]:


all_train['ScheduledArrTime'] = (pd.to_datetime(all_train['ScheduledArrTime'].astype(str).str.zfill(4), format='%H%M').dt.strftime('%H%M'))
all_train['ScheduledArrTime']=pd.to_datetime(all_train['ScheduledArrTime'], format='%H%M')


# In[180]:


all_train.head()


# In[181]:


all_train['dateInt']=all_train['Year'].astype(str) + all_train['Month'].astype(str).str.zfill(2)+ all_train['DayofMonth'].astype(str).str.zfill(2)
all_train['Date'] = pd.to_datetime(all_train['dateInt'], format='%Y%m%d')


# In[182]:


all_train.dtypes


# In[183]:


hourly_data.dtypes


# In[184]:


hourly_data.head(2)


# In[185]:


all_train.head(2)


# In[186]:


hourly_data=hourly_data.sort_values(by=['Time'])
all_train=all_train.sort_values(by=['ScheduledArrTime'])


# In[187]:


final_train=pd.merge_asof(all_train,hourly_data,left_on='ScheduledArrTime',right_on='Time',by=['Date','WeatherStationID'],tolerance=pd.Timedelta('60m'),direction='nearest')


# https://stackoverflow.com/questions/56633577/pandas-merge-asof-does-not-want-to-merge-on-pd-timedelta-giving-error-must-be-c

# In[188]:


final_train.shape


# In[189]:


final_train.head()


# In[190]:


final_train.drop(['Year_x','ScheduledArrTime','ActualArrivalTimeStamp','AirportID','ScheduledDepTime','Time','ScheduledTravelTime','dateInt','YearMonthDay','Year_y','Month_y',"Day"],axis=1,inplace=True)


# In[191]:


final_train.isnull().sum()


# In[192]:


final_train.dropna(axis=0,inplace=True)


# In[193]:


final_train.head()


# In[194]:


final_train.describe()


# In[195]:


final_train.info()


# In[196]:


final_train['TimeZone'] = final_train['TimeZone'].astype(str).astype(int)


# In[197]:


final_train.Visibility.value_counts()


# In[198]:


#remove the units 
final_train['Visibility'] = final_train['Visibility'].str.replace('SM','')


# In[199]:


final_train['WindSpeed'] = pd.to_numeric(final_train['WindSpeed'])


# In[200]:


final_train.WindDirection.value_counts()


# In[201]:


#replacing vrb with zero as it is categorical
#vrb means variable wind and its direction cannot be determined.
final_train['WindDirection'] = final_train['WindDirection'].apply(lambda x: 0 if x == 'VRB' else x)


# In[202]:


final_train['WindDirection'] = pd.to_numeric(final_train['WindDirection'])


# In[203]:


import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(15,15))  
dataplot = sns.heatmap(final_train.corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[204]:


sns.FacetGrid(final_train, hue = "FlightDelaystatus").map(sns.distplot, "Month_x").add_legend()
plt.show()


# The delay is happening more in winter season 

# In[205]:


final_train.drop(['FlightNumber','Destination','Origin','WeatherStationID','ActualDate','ActualTime','Actual','FlightDelayStatus','Date'],axis=1,inplace=True)


# In[206]:


final_train.head()


# In[207]:


final_train.GroundHeight.value_counts()


# # Check for outliers

# In[208]:


import seaborn as sns

sns.boxplot(final_train['GroundHeight'])


# In[209]:


final_train.GroundHeight.median()


# In[210]:


#replacing outliers with median

final_train['GroundHeight'] = np.where(final_train['GroundHeight'] > 2000,559, final_train['GroundHeight'])


# In[211]:


sns.boxplot(final_train['StationHeight'])


# In[212]:


final_train.StationHeight.median()


# In[213]:


final_train['StationHeight'] = np.where(final_train['StationHeight'] > 2000,596, final_train['StationHeight'])


# In[214]:


sns.boxplot(final_train['BarometerHeight'])


# In[215]:


final_train.BarometerHeight.median()


# In[216]:


final_train['BarometerHeight'] = np.where(final_train['BarometerHeight'] > 2000,562, final_train['BarometerHeight'])


# In[217]:


final_train.WindGustValue.hist()


# In[218]:


print(final_train['WindGustValue'].quantile(0.1))
print(final_train['WindGustValue'].quantile(0.95))


# In[219]:


#windgustvalue is usually less than 20 
final_train['WindGustValue'] = np.where(final_train['WindGustValue'] > 22,0, final_train['WindGustValue'])


# In[220]:


final_train['Visibility'] = final_train['Visibility'].astype(str).astype(float)


# In[221]:


sns.lineplot(data=final_train, x="Month_x", y="Visibility")


# In[222]:


final_train.head()


# # Considering test data and 2005 hourly data. 

# In[223]:


test=pd.read_csv("Test.csv")


# In[224]:


test.head()


# In[225]:


all_test=pd.merge(test,all_station,left_on="Destination",right_on="AirportID",how="left")


# In[226]:


all_test.head()


# In[227]:


all_test.Month.value_counts()


# In[228]:


mar_test_hr=pd.read_csv("200503hourly.txt")
jul_test_hr=pd.read_csv("200507hourly.txt")
sep_test_hr=pd.read_csv("200509hourly.txt")
nov_test_hr=pd.read_csv("200511hourly.txt")


# In[229]:


hour_test=[mar_test_hr,jul_test_hr,sep_test_hr,nov_test_hr]
hourly_data_test=pd.concat(hour_test)


# In[230]:


hourly_data_test.head()


# In[231]:


hourly_data_test.isnull().sum()/len(hourly_data_test)*100


# In[232]:


hourly_data_test.dropna(axis=0,inplace=True)


# In[233]:


hourly_data_test.drop_duplicates(inplace=True)


# In[234]:


hourly_data_test.shape


# In[235]:


hourly_data_test['YearMonthDay']=hourly_data_test.YearMonthDay.astype(str)
hourly_data_test['Year']=hourly_data_test.YearMonthDay.str[0:4]
hourly_data_test['Month']=hourly_data_test.YearMonthDay.str[4:6]
hourly_data_test['Day']=hourly_data_test.YearMonthDay.str[6:8]


# In[236]:


hourly_data_test['Date']=pd.to_datetime(hourly_data_test[["Year", "Month", "Day"]])


# In[237]:


hourly_data_test['Time'] = (pd.to_datetime(hourly_data_test['Time'].astype(str).str.zfill(4), format='%H%M').dt.strftime('%H:%M'))


# In[238]:


hourly_data_test['Time']=pd.to_datetime(hourly_data_test['Time'], format='%H:%M')


# In[239]:


hourly_data_test.drop(['Year','YearMonthDay'],axis=1,inplace=True)


# In[240]:


hourly_data_test.info()


# In[241]:


all_test.head()


# In[242]:


all_test.rename(columns = {'DayofMonth':'Day'}, inplace = True)


# In[243]:


all_test['Date']=pd.to_datetime(all_test[["Year", "Month", "Day"]])


# In[244]:


all_test.head()


# In[245]:


all_test['ScheduledArrTime'] = (pd.to_datetime(all_test['ScheduledArrTime'].astype(str).str.zfill(4), format='%H%M').dt.strftime('%H%M'))
all_test['ScheduledArrTime']=pd.to_datetime(all_test['ScheduledArrTime'], format='%H%M')


# In[246]:


hourly_data_test['Time']=pd.to_datetime(hourly_data_test['Time'], format='%H:%M')


# In[247]:


#hpd_data_test['Time']=pd.to_datetime(hpd_data_test['Time'], format='%H:%M')


# In[248]:


#hourly_data_test=pd.merge(hourly_data_test,hpd_data_test,on=['WeatherStationID','Date','Time'],how="left")


# In[249]:


hourly_data_test=hourly_data_test.sort_values(by=['Time'])
all_test=all_test.sort_values(by=['ScheduledArrTime'])


# In[250]:


final_test=pd.merge_asof(all_test,hourly_data_test,left_on='ScheduledArrTime',right_on='Time',by=['Date','WeatherStationID'],tolerance=pd.Timedelta('60m'),direction='nearest')


# In[251]:


final_test.shape


# In[252]:


final_test.head()


# In[253]:


final_test.isnull().sum()


# In[254]:


final_test.dropna(axis=0,inplace=True)


# In[255]:


final_test.drop_duplicates(inplace=True)


# In[256]:


final_test.shape


# In[257]:


final_test.drop(['Year','WeatherStationID','FlightNumber','Date','ScheduledDepTime','ScheduledArrTime','ScheduledTravelTime','Origin','Destination','AirportID','Time','SkyConditions'],axis=1,inplace=True)


# In[258]:


final_test.head()


# In[259]:


final_test['Visibility'] = final_test['Visibility'].str.replace('SM','')


# In[260]:


final_test['Visibility']= final_test['Visibility'].astype(str).astype(float)
final_test['TimeZone'] = final_test['TimeZone'].astype(str).astype(int)
final_test['WindSpeed']= final_test['WindSpeed'].astype(str).astype(float)


# In[261]:


final_test.WindDirection.value_counts()


# In[262]:


final_test['WindDirection'] = final_test['WindDirection'].apply(lambda x: 0 if x == 'VRB' else x)
final_test['WindDirection'] = pd.to_numeric(final_test['WindDirection'])


# In[263]:


final_test.head()


# In[264]:


import seaborn as sns

sns.boxplot(final_test['GroundHeight'])


# In[265]:


final_test.GroundHeight.median()


# In[266]:


final_test['GroundHeight'] = np.where(final_test['GroundHeight'] > 2000,559, final_test['GroundHeight'])


# In[267]:


sns.boxplot(final_test['StationHeight'])


# In[268]:


final_test.StationHeight.median()


# In[269]:


final_test['StationHeight'] = np.where(final_test['StationHeight'] > 2000,596, final_test['StationHeight'])


# In[270]:


sns.boxplot(final_test['BarometerHeight'])


# In[271]:


final_test.BarometerHeight.median()


# In[272]:


final_test['BarometerHeight'] = np.where(final_test['BarometerHeight'] > 2000,562, final_test['BarometerHeight'])


# In[273]:


final_test.WindGustValue.hist()


# In[274]:


print(final_test['WindGustValue'].quantile(0.1))
print(final_test['WindGustValue'].quantile(0.95))


# In[275]:


#windgustvalue is usually less than 20 
final_test['WindGustValue'] = np.where(final_test['WindGustValue'] > 23,0, final_test['WindGustValue'])


# In[276]:


final_test.drop(['Month_y','Day_y'],inplace=True,axis=1)


# In[277]:


final_test.rename(columns = {'Day_x':'DayofMonth'}, inplace = True)


# In[278]:


final_test.info()


# In[279]:


final_train.info()


# In[280]:


final_train.drop(['SkyConditions'],axis=1,inplace=True)


# In[281]:


count_classes=pd.value_counts(final_train['FlightDelaystatus'],sort=True)
count_classes.plot(kind='bar',rot=0)
plt.title("Delay status distribution")
plt.xlabel('Delay')
plt.ylabel('count')


# In[282]:


final_train_X = final_train.drop(columns=['FlightDelaystatus']) 
final_train_y = final_train['FlightDelaystatus']


# In[ ]:





# In[283]:


import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, recall_score, precision_score, roc_curve, auc


# In[284]:


X_train,X_test,y_train, y_test = train_test_split(final_train_X, final_train_y, test_size = 0.2, random_state = 32)


# # Using Support Vector Machines

# In[285]:


from sklearn.svm import SVC
linear_svm = SVC(kernel='linear')
linear_svm = linear_svm.fit(X=X_train, y= y_train)
linear_svm.support_vectors_


# In[286]:


train_predictions = linear_svm.predict(X_train)
test_predictions = linear_svm.predict(X_test)


# In[287]:


print("Accuracy on training set: {:.3f}".format(linear_svm.score(X_train, y_train)*100))
print("Accuracy on testing set: {:.3f}".format(linear_svm.score(X_test, y_test)*100))


# In[288]:


print(classification_report(y_train, train_predictions))


# In[289]:


predictions = linear_svm.predict(final_test)


# In[ ]:


# final_1 = test.iloc[:,0:1]
# final_2 = pd.DataFrame(predictions,columns = ['predictions'])
# result = pd.concat([final_1, final_2], axis=1)

# result.to_csv('SVC_3.csv',index=False)


# # Using RandomForestClassifier

# In[290]:


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

rf1=RandomForestClassifier(max_depth=10,class_weight='balanced')
rf1.fit(X_train, y_train)


# In[291]:


#predict with train and test sample data
y_train_pred = rf1.predict(X_train)
y_test_pred = rf1.predict(X_test)


# In[292]:


#Check Accuracy
print("Accuracy on training set: {:.3f}".format(rf1.score(X_train, y_train)))
print("Accuracy on testing set: {:.3f}".format(rf1.score(X_test, y_test)))


# In[293]:


#predicting with test data
predictions_rfc = rf1.predict(final_test)


# In[296]:


print(classification_report(y_train,y_train_pred))


# In[ ]:


final_1 = test.iloc[:,0:1]
final_2 = pd.DataFrame(predictions,columns = ['predictions'])
result = pd.concat([final_1, final_2], axis=1)

result.to_csv('Output.csv',index=False)


# # Using SMOTE to avoid imbalance in dataset

# In[297]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
sm.fit_resample
X_res,y_res=sm.fit_resample(final_train_X,final_train_y)


# In[298]:


from sklearn.ensemble import RandomForestClassifier

rf1 = RandomForestClassifier(max_depth=10,class_weight='balanced')
rf1.fit(X_res, y_res)


# In[299]:


#predict with train and test sample data
y_train_pred = rf1.predict(final_test) 


# In[300]:


#Check Accuracy
print("Accuracy on training set: {:.3f}".format(rf1.score(X_res, y_res)))


# In[301]:


#predicting with test data
predictions_rfc = rf1.predict(final_test)
print(predictions_rfc)


# In[302]:


print(classification_report(predictions_rfc,y_train_pred))


# In[ ]:


#save to .csv file
final_1 = test.iloc[:,0:1]
final_2 = pd.DataFrame(predictions_rfc,columns = ['predictions'])
result = pd.concat([final_1, final_2], axis=1)

result.to_csv('RFC20.csv',index=False)


# # Using GridSearch for Hyperparameter Tuning

# In[303]:


from sklearn.model_selection import GridSearchCV


# In[304]:


parameters={'n_estimators':(10,30,50,70,90,100),
           'criterion':('gini','entropy'),
           'max_depth':(3,5,7,9,10),
           'max_features':('auto','sqrt'),
           'min_samples_split':(2,4,6)
            }


# In[305]:


RF_grid=GridSearchCV(RandomForestClassifier(n_jobs=-1,oob_score=False),param_grid=parameters,cv=3,verbose=True)  


# In[306]:


RF_grid_model=RF_grid.fit(X_res,y_res)


# In[307]:


RF_grid_model.best_estimator_


# In[308]:


RF_grid_model.best_score_


# In[309]:


RF_model=RandomForestClassifier(max_depth=10, min_samples_split=4, n_estimators=50,
                       n_jobs=-1)
RF_model.fit(X_res, y_res)


# In[310]:


#predict with train and test sample data
y_train_pred = RF_model.predict(X_res)


# In[311]:


#Check Accuracy
print("Accuracy on training set: {:.3f}".format(RF_model.score(X_res, y_res)))


# In[312]:


#predicting with test data
predictions_test = RF_model.predict(final_test)


# In[313]:


print(classification_report(y_res,y_train_pred))


# In[ ]:


# save to .csv file
final_1 = test.iloc[:,0:1]
final_2 = pd.DataFrame(predictions_rfc,columns = ['predictions'])
result = pd.concat([final_1, final_2], axis=1)

result.to_csv('RFC24.csv',index=False)


# # Using Decision Tree

# In[314]:


from sklearn.tree import DecisionTreeClassifier


# In[315]:


clf_gini = DecisionTreeClassifier(criterion = "entropy",random_state = 100,max_depth=3, min_samples_leaf=5)
  
clf_gini.fit(X_train, y_train)


# In[316]:


#predict with train and test sample data
y_train_pred = clf_gini.predict(X_train)
y_test_pred = clf_gini.predict(X_test)


# In[317]:


#Check Accuracy
print("Accuracy on training set: {:.3f}".format(clf_gini.score(X_train, y_train)))
print("Accuracy on testing set: {:.3f}".format(clf_gini.score(X_test, y_test)))


# In[318]:


#predicting with test data
predictions_dt = clf_gini.predict(final_test)
print(predictions_dt)


# In[319]:


print(classification_report(y_train,y_train_pred))


# # Using AdaBoost

# In[320]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
abc=AdaBoostClassifier(n_estimators=100, random_state=0)


# In[321]:


abc.fit(X_res,y_res)


# In[322]:


#predict with train and test sample data
y_train_pred = abc.predict(X_res)


# In[323]:


#Check Accuracy
print("Accuracy on training set: {:.3f}".format(abc.score(X_res, y_res)))


# In[324]:


#predicting with test data
predictions_ab = abc.predict(final_test)
print(predictions_ab)


# In[325]:


print(classification_report(y_res,y_train_pred))


# In[ ]:


#save to .csv file
final_1 = test.iloc[:,0:1]
final_2 = pd.DataFrame(predictions_ab,columns = ['predictions'])
result = pd.concat([final_1, final_2], axis=1)

result.to_csv('AB2.csv',index=False)


# # Thank You

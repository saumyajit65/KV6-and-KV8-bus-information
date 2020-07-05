import pandas as pd
import numpy as np
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
import matplotlib.pyplot as pyplot
import math as m
from tensorflow import keras
import random
import json
import re
import os
import seaborn as sns
from sklearn import linear_model, preprocessing
from random import seed
from random import choice
from collections import Counter
import xlsxwriter
import gzip
from gzip import GzipFile
import zmq
import xml.etree.cElementTree as ET
from datetime import datetime
from datetime import date
from time import *
import csv
from io import BytesIO
import csv
from builtins import enumerate
import requests
import polyline

context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://pubsub.besteffort.ndovloket.nl:7817")
#subscriber.setsockopt(zmq.SUBSCRIBE, "/RIG/KV6posinfo")
subscriber.setsockopt(zmq.SUBSCRIBE, b"")
a= []
count = 0
while True:
    multipart = subscriber.recv_multipart()
    address = multipart[0]
    contents = b"".join(multipart[1:])

    try:
        contents = [GzipFile('','r',0,BytesIO(contents)).read()]
    except:
        raise
        print('NOT ', contents)
    for i in contents:
        a.append(i)
        count = count + 1
        print(a)
        # if not a[count].startswith("b'/GOVI/KV8passtimes"):
        #    a[count-1] = a[count-1] + a[count]
        #if (count > 0):
        np.savetxt("C:/Users/Saumyajit/PycharmProjects/saumyajitAIML4/kv8rough.csv", a, delimiter="", fmt='%s')
        if count > 450:
            break
    if count > 449:
        break
subscriber.close()
context.term() #terminate the context


data = pd.read_csv(r'C:/Users/Saumyajit/PycharmProjects/saumyajitAIML4/kv8rough.csv', error_bad_lines=False, header=None)
data1 = data.iloc[:,0]
#pd.set_option("display.max_rows", None, "display.max_columns", None)
data2 = data1.str.rsplit("|", expand=True)
data3= data2.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
                   32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,
                   56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74], 1)
data4= data3.values.tolist()
labels = ['OperationDate', 'LinePlanningNumber'	,'JourneyNumber',	'FortifyOrderNumber',	'UserStopOrderNumber',	'UserStopCode',	'LocalServiceLevelCode',
          'JourneyPatternCode',	'LineDirection',	'LastUpdateTimeStamp',	'DestinationCode',	'IsTimingStop',	'ExpectedArrivalTime',	'ExpectedDepartureTime',
          'TripStopStatus',	'MessageContent',	'MessageType',	'SideCode',	'NumberOfCoaches',	'WheelChairAccessible',	'OperatorCode',	'ReasonType',
          'SubReasonType',	'ReasonContent',	'AdviceType', 'SubAdviceType',	'AdviceContent',	'TimingPointDataOwnerCode',
          'TimingPointCode',	'JourneyStopType',	'TargetArrivalTime',	'TargetDepartureTime',	'RecordedArrivalTime',
          'RecordedDepartureTime',	'DetectedUserStopCode',	'DistanceSinceDetectedUserStop',	'Detected_RD_X',	'Detected_RD_Y',
          'VehicleNumber',	'BlockCode',	'LineVeTagNumber',	'VejoJourneyNumber',	'VehicleJourneyType',	'VejoBlockNumCode',
          'JourneyModificationType',	'VejoDepartureTime',	'VejoArrivalTime',	'VejoTripStatusType',	'ExtraJourney',
          'CancelledJourney',	'ShowCancelledTrip',	'ShowFlexibleTrip',	'Monitored',	'MonitoringError',	'ExtraCall',	'CancelledCall',
          'ShowCancelledStop',	'AimedQuayRef',	'ExpectedQuayRef',	'ActualQuayRef',	'Occupancy',	'LineDestIcon',	'LineDestColor',
          'LineDestTextColor\r\nSYNTUS']
d = []
row = -1
column =0
for i in range(len(data4)):
    e=[]
    for j in range(len(data4[i])):
        if data4[i][j] != None:
            e.append(data4[i][j])
            column = column + 1
            if column > 63:
                d.append(e)
                e = []
                column = 0
                my_df = pd.DataFrame(d)
                

    column  = 0
my_df.to_csv('kv8finaldata.csv', index=False, header=False)

# For KV 6 messages
context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://pubsub.besteffort.ndovloket.nl:7658")
subscriber.setsockopt(zmq.SUBSCRIBE, b"/RIG/KV6posinfo")
#subscriber.setsockopt(zmq.SUBSCRIBE, b"")
a= []
count = 0
while True:
    multipart = subscriber.recv_multipart()
    address = multipart[0]
    contents = b"".join(multipart[1:])

    try:
        contents = [GzipFile('','r',0,BytesIO(contents)).read()]
        print(contents)
    except:
        raise
        print('NOT ', contents)
    for i in contents:
        a.append(i)
        count = count + 1

        # if not a[count].startswith("b'/GOVI/KV8passtimes"):
        #    a[count-1] = a[count-1] + a[count]
        #if (count > 0):
        np.savetxt("C:/Users/Saumyajit/PycharmProjects/saumyajitAIML4/kv6rough.csv", a, delimiter="", fmt='%s')
        if count > 450:
            break
    if count > 449:
        break
subscriber.close()
context.term() #terminate the context

h1 = []
count = 0
a1 = [];    b1 = [];    c1 = [];    d1 = [];    e1 = [];    f1 = [];    g1 = [];
data = pd.read_csv(r'C:/Users/Saumyajit/PycharmProjects/saumyajitAIML4/kv6rough.csv', error_bad_lines=False, header=None)
data1 = data.iloc[:,0]

for longtext in data1:
    c= []
    str2 = "http://bison.connekt.nl/tmi8/kv6/msg"
    #ss= longtext.find(str2)
    if str2 in longtext:
        pattern3 = "userstopcode>(.*?)</tmi8:userstopcode"
        pattern4 = "vehiclenumber>(.*?)</tmi8:vehiclenumber>"
        #pattern5 = "<tmi8:distancesincelastuserstop(.*?)</tmi8:distancesincelastuserstop"
        pattern6 = "tmi8:rd-x>(.*?)</tmi8/"
        pattern7 = "tmi8:rd-y>(.*?)</tmi8/"
        c= re.findall(pattern3, longtext)
        if not any(c):
            continue
        else:
            c1.append(c)


for longtext in data1:
    d= []
    str2 = "http://bison.connekt.nl/tmi8/kv6/msg"
    #ss= longtext.find(str2)
    if str2 in longtext:
        pattern4 = "vehiclenumber>(.*?)</tmi8:vehiclenumber>"
        d= re.findall(pattern4, longtext)
        if not any(d):
            continue
        else:
            d1.append(d)

for longtext in data1:
    e= []
    str2 = "http://bison.connekt.nl/tmi8/kv6/msg"
    if str2 in longtext:
        pattern6 = "tmi8:rd-x>(.*?)</tmi8:rd-x"
        e= re.findall(pattern6, longtext)
        if not any(e):
            continue
        else:
            e1.append(e)

for longtext in data1:
    f= []
    str2 = "http://bison.connekt.nl/tmi8/kv6/msg"
    #ss= longtext.find(str2)
    if str2 in longtext:
        pattern7 = "tmi8:rd-y>(.*?)</tmi8:rd-y>"
        f= re.findall(pattern7, longtext)
        if not any(f):
            continue
        else:
            f1.append(f)
dc = pd.DataFrame(c1)
dc1 = dc.stack()
dd = pd.DataFrame(d1)
dd1 = dd.stack()
de = pd.DataFrame(e1)
de1 = de.stack()
df = pd.DataFrame(f1)
df1 = df.stack()
output1 = pd.concat([dc1,dd1,de1,df1],axis=1)

output1.to_csv('kv6finaldata.csv', index=False, header=False)

# final data concat
data13= pd.read_csv("kv8finaldata.csv")
data14 = pd.read_csv("kv6finaldata.csv")
operationdate = data13.iloc[:,0]
operationdate1 = pd.DataFrame(operationdate)
journeynumber = data13.iloc[:,2]
journeynumber1 = pd.DataFrame(journeynumber)
userstopcode = data13.iloc[:,5]
userstopcode1 = pd.DataFrame(userstopcode)
JourneyPatternCode = data13.iloc[:,7]
JourneyPatternCode1 = pd.DataFrame(JourneyPatternCode)
ExpectedArrivalTime = data13.iloc[:,12]
ExpectedArrivalTime1 = pd.DataFrame(ExpectedArrivalTime)
ExpectedDepartureTime = data13.iloc[:,13]
ExpectedDepartureTime1 = pd.DataFrame(ExpectedDepartureTime)
TimingPointCode = data13.iloc[:,28]
TimingPointCode1 = pd.DataFrame(TimingPointCode)
JourneyStopType = data13.iloc[:,29]
JourneyStopType1 = pd.DataFrame(JourneyStopType)
TargetArrivalTime = data13.iloc[:,30]
TargetArrivalTime1 = pd.DataFrame(TargetArrivalTime)
TargetDepartureTime= data13.iloc[:,31]
TargetDepartureTime1 = pd.DataFrame(TargetDepartureTime)
RecordedArrivalTime= data13.iloc[:,32]
RecordedArrivalTime1 = pd.DataFrame(RecordedArrivalTime)
RecordedDepartureTime = data13.iloc[:,33]
RecordedDepartureTime1 = pd.DataFrame(RecordedDepartureTime)
DetectedUserStopCode= data13.iloc[:,34]
DetectedUserStopCode1 = pd.DataFrame(DetectedUserStopCode)
DistanceSinceDetectedUserStop = data13.iloc[:,35]
DistanceSinceDetectedUserStop1 = pd.DataFrame(DistanceSinceDetectedUserStop)
VehicleNumber= data13.iloc[:,38]
VehicleNumber1 = pd.DataFrame(VehicleNumber)
VejoDepartureTime = data13.iloc[:,45]
VejoDepartureTime1 = pd.DataFrame(VejoDepartureTime)
VejoArrivalTime= data13.iloc[:,46]
VejoArrivalTime1 = pd.DataFrame(VejoArrivalTime)
Messagefrom= data13.iloc[:,63]
Messagefrom1 = pd.DataFrame(Messagefrom)
kv6vehicleno = data14.iloc[:,0]
kv6vehicleno1 = pd.DataFrame(kv6vehicleno)
kv6userstopcode = data14.iloc[:,1]
kv6userstopcode1 = pd.DataFrame(kv6userstopcode)
kv6rdx = data14.iloc[:,2]
kv6rdx1 = pd.DataFrame(kv6rdx)
kv6rdy = data14.iloc[:,3]
kv6rdy1 = pd.DataFrame(kv6rdy)
output2 = pd.concat([operationdate1,journeynumber1,userstopcode1,JourneyPatternCode1,ExpectedArrivalTime1,ExpectedDepartureTime1,TimingPointCode1,
                     JourneyStopType1,TargetArrivalTime1,TargetDepartureTime1,RecordedArrivalTime1,RecordedDepartureTime1,
                     DetectedUserStopCode1, DistanceSinceDetectedUserStop1, VehicleNumber1,VejoDepartureTime1,
                     VejoArrivalTime1,Messagefrom1, kv6vehicleno1, kv6userstopcode1, kv6rdx1, kv6rdy1],axis=1)
output2.to_csv('KV6andkv8finalmerged.csv', index=False, header=False)

# For converting rdx and rdy to lat and long
class RDWGS84Converter(object):
    x0 = 155000
    y0 = 463000
    phi0 = 52.15517440
    lam0 = 5.38720621

    # Coefficients or the conversion from RD to WGS84
    Kp = [0, 2, 0, 2, 0, 2, 1, 4, 2, 4, 1]
    Kq = [1, 0, 2, 1, 3, 2, 0, 0, 3, 1, 1]
    Kpq = [3235.65389, -32.58297, -0.24750, -0.84978, -0.06550, -0.01709, -0.00738, 0.00530, -0.00039,
           0.00033, -0.00012]

    Lp = [1, 1, 1, 3, 1, 3, 0, 3, 1, 0, 2, 5]
    Lq = [0, 1, 2, 0, 3, 1, 1, 2, 4, 2, 0, 0]
    Lpq = [5260.52916, 105.94684, 2.45656, -0.81885, 0.05594, -0.05607, 0.01199, -0.00256, 0.00128, 0.00022,
           -0.00022, 0.00026]

    # Converts RD coordinates into WGS84 coordinates
    def from_rd1(self, x: int, y: int) -> list:
        dx = 1E-5 * (x - self.x0)
        dy = 1E-5 * (y - self.y0)
        latitude = self.phi0 + sum([v * dx ** self.Kp[i] * dy ** self.Kq[i] for i, v in enumerate(self.Kpq)]) / 3600
        longitude = self.lam0 + sum([v * dx ** self.Lp[i] * dy ** self.Lq[i] for i, v in enumerate(self.Lpq)]) / 3600

        return latitude
    def from_rd2(self, x: int, y: int) -> list:
        dx = 1E-5 * (x - self.x0)
        dy = 1E-5 * (y - self.y0)
        latitude = self.phi0 + sum([v * dx ** self.Kp[i] * dy ** self.Kq[i] for i, v in enumerate(self.Kpq)]) / 3600
        longitude = self.lam0 + sum([v * dx ** self.Lp[i] * dy ** self.Lq[i] for i, v in enumerate(self.Lpq)]) / 3600

        return  longitude

data14 = pd.read_csv("KV6andkv8finalmerged.csv")
count123 = -1
count124 = -1
rdx = data14.iloc[:,20]
rdy = data14.iloc[:,21]
rdx2=[0 for i in range(len(rdx))]
rdy2 = [0 for i in range(len(rdy))]
r1 = RDWGS84Converter()
for j in rdx:
    count123 = count123 + 1
    if not any(rdx):
        continue
    else:
        count124 = count124 +1
        rdx2[count124] = r1.from_rd1(rdx[count123], rdy[count123])
        rdy2[count124] = r1.from_rd2(rdx[count123],rdy[count123])
        data14['Latitude'] = rdx2 # it is also called WGS84 or EPSG 4326 system of referencing
        data14['Longitude'] = rdy2
data14.to_csv("KVfinal.csv", index=False, header=False) # mode a increases the size substantially



def match(orign, destination):
    data = pd.read_csv("ODmatrix2.csv")
    longi = data.iloc[:, 2]
    lati = data.iloc[:, 3]
    timecode = data.iloc[:, 0]
    timecode1 = timecode.values.tolist()
    position_orign_long = longi[timecode1.index(orign)]
    position_orign_lat = lati[timecode1.index(orign)]
    position_dest_long = longi[timecode1.index(destination)]
    position_dest_lat = lati[timecode1.index(destination)]
    loc = "{},{};{},{}".format(position_orign_long, position_orign_lat, position_dest_long, position_dest_lat)
    url = "http://router.project-osrm.org/route/v1/driving/"
    r = requests.get(url + loc)
    if r.status_code != 200:
        return {}
    res = r.json()
    routes = polyline.decode(res['routes'][0]['geometry'])
    start_point = [res['waypoints'][0]['location'][1], res['waypoints'][0]['location'][0]]
    end_point = [res['waypoints'][1]['location'][1], res['waypoints'][1]['location'][0]]
    distance = res['routes'][0]['distance']
    out = distance
    return out
print(match(31000099,31000103))


"""
distlist = []
data = pd.read_csv("ODmatrix2.csv")
longi = data.iloc[:, 2]
lati = data.iloc[:, 3]
data23 = pd.read_csv("KVfinal.csv")

labels = ['operationdate','journeynumber1','userstopcode1','JourneyPatternCode1','ExpectedArrivalTime1','ExpectedDepartureTime1','TimingPointCode1',
                     'JourneyStopType1','TargetArrivalTime1','TargetDepartureTime1','RecordedArrivalTime1','RecordedDepartureTime1',
                     'DetectedUserStopCode1', 'DistanceSinceDetectedUserStop1', 'VehicleNumber1','VejoDepartureTime1',
                     'VejoArrivalTime1','Messagefrom1', 'kv6vehicleno1', 'kv6userstopcode1', 'kv6rdx1', 'kv6rdy1', 'lati', 'longi']
data23.columns = labels
data14 = data23.set_index('Messagefrom1').filter(like='RET', axis=0)
data141 = data14[~data14['TimingPointCode1'].str.contains('H', na=False)] # for filtering out columns having h as a character
data15 = data141.sort_values(by='journeynumber1') # each journey number is unique for a vehicle
data16 = data15.journeynumber1.unique()
data17 = data15[(data15.journeynumber1 == data16[0])]
data18 = data17.sort_values(by='ExpectedArrivalTime1')
#pd.set_option("display.max_rows", None, "display.max_columns", None)
for column in data18['TimingPointCode1']:
    print(column)






for i in timecode1:
    row_list = []
    for j in timecode1:
        row_list.append(match(i,j))
    distlist.append(row_list)
print(distlist)
"""
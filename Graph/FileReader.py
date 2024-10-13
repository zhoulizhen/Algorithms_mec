import pandas as pd
def read_file(usrnum,clsnum):

    cls = pd.read_csv('../Dataset/edge-servers/site.csv',usecols=['CL_ID','LATITUDE','LONGITUDE'], nrows=clsnum)

    ues = pd.read_csv('../Dataset/users/users-melbcbd-generated.csv', usecols=['User_ID', 'Latitude', 'Longitude'], nrows=usrnum)

    return cls,ues


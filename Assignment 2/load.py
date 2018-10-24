
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd

print(__doc__)
if __name__ == "__main__":
    
    
    # use course number as random_seed
    random_seed = 755
    
    def writeData(src, train, test):
        train.reset_index().to_csv(src + '/train.csv', index=False)
        test.reset_index().to_csv(src + '/test.csv', index=False)
    
    
    # Traffic Congestion Data Preparation
    dat2 = pd.read_csv("data/Traffic_flow/traffic_flow_data.csv")
    train, test = train_test_split(dat2, test_size=0.1, random_state=random_seed)
    writeData('data/Traffic_flow', train, test)
    
    train['Segment23_(t+1)'].describe()
    #count    6750.000000
    #mean      301.378222
    #std       198.898209
    #min         0.000000
    #25%        93.000000
    #50%       327.000000
    #75%       482.000000
    #max      1350.000000
    #Name: Segment23_(t+1), dtype: float64
    test['Segment23_(t+1)'].describe()
    #count    750.000000
    #mean     313.038667
    #std      195.452943
    #min        0.000000
    #25%      120.000000
    #50%      359.500000
    #75%      485.000000
    #max      815.000000
    #Name: Segment23_(t+1), dtype: float64
    
    # Occupancy Sensor Data Preparation
    dat3 = pd.read_csv("data/Occupancy_sensor/occupancy_sensor_data.csv")
    train, test = train_test_split(dat3, test_size=0.1, 
                                   random_state=random_seed,
                                   stratify = dat3['Occupancy'])
    writeData('data/Occupancy_sensor', train, test)
    print("Occupancy Target Variable Distribution")
    print("Train")
    print(train.Occupancy.value_counts())
    #0    12120
    #1     3288
    #Name: Occupancy, dtype: int64
    print("Test")
    print(test.Occupancy.value_counts())
    #0    1347
    #1     365
    #Name: Occupancy, dtype: int64
    
    
    # LandSat Data Preparation
    # In each line of data the four spectral values for the top-left pixel are given first 
    # followed by the four spectral values for the top-middle pixel 
    # and then those for the top-right pixel, so on with the pixels read 
    # out in sequence left-to-right and top-to-bottom. 
    # Thus, the four spectral values for the central pixel are given by attributes 17,18,19 and 20. 
    # If you like you use only these four attributes, 
    # while ignoring the others. 
    # This avoids the problem which arises when a 3x3 neighbourhood straddles a boundary.
    location = ["top-left", "top-mid", "top-right",
                "mid-left", "mid-mid", "mid-right",
                "bottom-left","bottom-mid", "bottom-right"]
    location2 = [item+"_"+ str(num) for item in location for num in [1,2,3,4]]
    dat4 = pd.read_csv("data/Landsat/lantsat.csv", names=location2+["label"])
    dat4.label.value_counts() # classification
    dat4.iloc[:,16:20] # core
    train, test = train_test_split(dat4, test_size=0.1, stratify=dat4.label, 
                                   random_state=random_seed)
    writeData('data/Landsat', train, test)
    print("Landsat Target Variable Distribution")
    print("Train")
    print(train.label.value_counts())
    #1    1153
    #3    1145
    #2     619
    #5     580
    #4     559
    #Name: label, dtype: int64
    print("Test")
    print(test.label.value_counts())
    #7    149
    #1    128
    #3    127
    #2     69
    #5     65
    #4     62
    #Name: label, dtype: int64

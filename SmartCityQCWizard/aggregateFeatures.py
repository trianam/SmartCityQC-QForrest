import pandas as pd

inputDatasetFile = "final_ds.csv"
outputDatasetFile = "final_ds_agg.csv"

df = pd.read_csv(inputDatasetFile)

df['AttendanceAreaTot'] = df['AttendanceArea1'] + df['AttendanceArea2'] + df['AttendanceArea3'] + df['AttendanceArea4'] + df['AttendanceArea5'] + df['AttendanceArea6'] + df['AttendanceArea7'] + df['AttendanceArea8'] + df['AttendanceArea9'] + df['AttendanceArea10'] + df['AttendanceArea11'] + df['AttendanceArea12']
df['PM2.5_avg'] = (df['PM2.5_Sensor1'] + df['PM2.5_Sensor2'] + df['PM2.5_Sensor3']) / 3
df['PM10_avg'] = (df['PM10_Sensor1'] + df['PM10_Sensor2'] + df['PM10_Sensor3']) / 3

df.to_csv(outputDatasetFile, index=False)

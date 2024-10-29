import runSVM
import runQSVMpl
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    datasetFile = "final_ds.csv"
    xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5',
                'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10',
                'AttendanceArea11', 'AttendanceArea12', 'PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3',
                'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3', 'hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg',
                'clouds_all', 'max_temp', 'min_temp', 'ave_temp']
    yColumns = ['cod_weather']
    windowSizeX = 2
    splitSeed = 42
    usePrecomputedKernel = True
    parallelizeKernel = True
    singleSplit = True
    # circuit = "amplitude"
    returnMF1 = True


    classicKernels = ["linear", "poly2", "poly4", "rbf", "sigmoid"]
    # classicKernels = []
    # quantumKernels = ["amplitude", "ampHad"]
    quantumKernels = ["amplitude"]
    wList = list(range(1, 25))
    # wList = [1] + list(range(2,25,2))
    # wList = [1,2,4]
    # wList = [1] + list(range(2,22,2))
    ySvm = {k:[] for k in classicKernels}
    yQsvm = {k:[] for k in quantumKernels}

    for windowSizeY in tqdm(wList):
        for kernel in classicKernels:
            ySvm[kernel].append(runSVM.run(datasetFile, xColumns, yColumns, windowSizeX, windowSizeY, singleSplit, returnMF1, kernel))
        for circuit in quantumKernels:
            yQsvm[circuit].append(runQSVMpl.run(datasetFile, xColumns, yColumns, windowSizeX, windowSizeY, splitSeed, usePrecomputedKernel, parallelizeKernel, singleSplit, circuit, returnMF1))

    plt.figure(figsize=(8,5))
    plt.title(f"Macro F1 score, single split, input {windowSizeX}h")

    for kernel in classicKernels:
        plt.plot(wList, ySvm[kernel], label=f"SVM {kernel}", linestyle=':', marker='.', fillstyle='none')
    for circuit in quantumKernels:
        plt.plot(wList, yQsvm[circuit], label=f"Q-SVM  {circuit}", linestyle='-', marker='o', fillstyle='full')
    plt.legend()
    plt.ylabel('Macro F1 score')
    plt.xlabel('Prediction window size')
    plt.xticks(wList)
    # plt.savefig('plot_svm-qsvm-wSizeY.pdf', bbox_inches='tight')
    plt.savefig('plot_svm-qsvm-wSizeY.png', bbox_inches='tight')
    plt.show(block=True)


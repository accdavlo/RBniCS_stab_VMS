import numpy as np
import ezyrb
from ezyrb import ReducedOrderModel as ROM
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler

train_NN = True
predict_phase = True

folder = "out_cylinder_test_spalding_gio_P2_N_100"

input_training_file = folder+"/training_input"
output_training_file = folder+"/training_output"

input_test_file = folder+"/training_input"
output_test_file = folder+"/training_test_output"

## Fake database, only to save the trained networks
db = ezyrb.Database()
reduction = ezyrb.POD()

component_input = "u"
component_output = "tau"
ROMs = dict()

NPOD_input = 30
NPOD_output = 30


def preprocess_data_time_0(input_data, output_data, output_data2 = None):
    # removing "time=0" lines
    input_data = input_data.deepcopy()
    output_data = output_data.deepcopy()
    output_data2 = output_data2.deepcopy()
    N_data = input_data.shape[0]
    i=0
    while (i<N_data):
        if abs(input_data[i,0]) <1e-16: #time is 0
            N_data=N_data-1
            # delete i-th row from input and output
            input_data = np.delete(input_data, i,0)
            output_data = np.delete(output_data, i,0)
            if output_data2 is not None:
                output_data2 = np.delete(output_data2, i,0)
        else:
            i=i+1
    return input_data, output_data, output_data2



if train_NN:
    time_data = np.load(input_training_file+".npy")
    input_data = np.load(output_training_file+"_"+component_input+".npy")
    output_data = np.load(output_training_file+"_"+component_output+".npy")
    
    time_data, input_data, output_data = preprocess_data_time_0(time_data, input_data,output_data)
    
    (N_train, N_param) = input_data.shape

    input_data_train = input_data[:N_train//2,:]
    output_data_train = output_data[:N_train//2,:]

    input_data_test = input_data[N_train//2:,:]
    output_data_test = output_data[N_train//2:,:]

    approximation = ezyrb.ANN([100, 100, 100, 100], nn.Tanh(), [20000, 1e-6])
    # approximation = ezyrb.RBF()

    scaler_input  = MinMaxScaler()
    scaler_output = MinMaxScaler()
    db.scaler_parameters = scaler_input
    ROM = ROM(db, reduction, approximation, scaler_red = scaler_output)
    if ROM.scaler_red:
        #scaling all outputs
        output_data_train = ROM.scaler_red.fit_transform(output_data_train)
    if ROM.database.scaler_parameters:
        #scaling all inputs
        input_data_train = ROM.database.scaler_parameters.fit_transform(input_data_train)


    ROM.approximation.fit(input_data_train, output_data_train)

    ROM.save(folder+"/ezyrb_"+component_input+"_to_"+component_output+".rom",\
            save_db=True,\
            save_reduction = False, save_approx = True)

    if ROM.database.scaler_parameters:
        #scaling all inputs
        input_data_test = ROM.database.scaler_parameters.fit_transform(input_data_test)

    output_data_predict_test = ROM.approximation.predict(input_data_test)
    if ROM.scaler_red:
        #scaling all outputs
        output_data_test_transformed = ROM.scaler_red.transform(output_data_test)
        print("Test error rescaled")
        error_test = np.mean(output_data_test_transformed-output_data_predict_test)
        print(error_test)

        output_data_predict_test = ROM.scaler_red.inverse_transform(output_data_predict_test)


    print("Test error")
    error_test = np.mean(output_data_test-output_data_predict_test)
    print(error_test)
    plt.plot(output_data_test[:,:10],"-")
    plt.plot(output_data_predict_test[:,:10],"--")
    plt.show()
        

        



# if predict_phase:
#     for comp in components:
#         input_data_predict = np.load(input_test_file+".npy")
#         ROMs[comp] = ROM.load(folder+"/ezyrb_"+comp+".rom")

#         output_data_predict = ROMs[comp].approximation.predict(input_data_predict)

#         np.save(output_test_file+"_"+comp+".npy", output_data_predict)


# if predict_phase:
#     for i in range(10):
#         for comp in components:
#             simul_folder = folder+"/param_%03d"%i
#             input_test_file = simul_folder+"/training_input"
#             input_data_predict = np.load(input_test_file+".npy")
#             ROMs[comp] = ROM.load(folder+"/ezyrb_"+comp+".rom")

#             output_data_predict = ROMs[comp].approximation.predict(input_data_predict)
#             output_test_file = simul_folder+"/test_output"
#             np.save(output_test_file+"_"+comp+".npy", output_data_predict)



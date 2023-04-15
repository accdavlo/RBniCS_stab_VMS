import numpy as np
import ezyrb
from ezyrb import ReducedOrderModel as ROM
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler

train_NN = True
predict_phase = False

folder = "out_cylinder_test_spalding_gio_P2_N_100"

input_training_file = folder+"/training_input"
output_training_file = folder+"/training_output"

input_test_file = folder+"/param_trial_RB/training_input"
output_test_file = folder+"/param_trial_RB/training_test_output"

## Fake database, only to save the trained networks
db = ezyrb.Database()
reduction = ezyrb.POD()

components = ["p"]#["tau"]#,"p","tau"]
ROMs = dict()


def preprocess_data_time_0(input_data, output_data):
    # removing "time=0" lines
    N_data = input_data.shape[0]
    i=0
    while (i<N_data):
        if abs(input_data[i,0]) <1e-16: #time is 0
            N_data=N_data-1
            # delete i-th row from input and output
            input_data = np.delete(input_data, i,0)
            output_data = np.delete(output_data, i,0)
        else:
            i=i+1
    return input_data, output_data



if train_NN:
    for comp in components:
        print(f"train component {comp}")
        input_data = np.load(input_training_file+".npy")
        output_data = np.load(output_training_file+"_"+comp+".npy")
        input_data, output_data = preprocess_data_time_0(input_data,output_data)
        
        (N_train, N_param) = input_data.shape

        input_data_train = input_data[:N_train//2,:]
        output_data_train = output_data[:N_train//2,:]

        input_data_test = input_data[N_train//2:,:]
        output_data_test = output_data[N_train//2:,:]

        NPOD = output_data.shape[1]

        approximation = ezyrb.ANN([100, 100], nn.Tanh(), [20000, 1e-3])
        # approximation = ezyrb.RBF()

        scaler = MinMaxScaler()

        ROMs[comp] = ROM(db,reduction, approximation, scaler_red = scaler)
        if ROMs[comp].scaler_red:
            #scaling all outputs
            output_data_train = ROMs[comp].scaler_red.fit_transform(output_data_train)
        ROMs[comp].approximation.fit(input_data_train, output_data_train)

        ROMs[comp].save(folder+"/ezyrb_"+comp+".rom", save_db=False,\
                        save_reduction = False, save_approx = True)

        output_data_predict_test = np.atleast_2d(ROMs[comp].approximation.predict(input_data_test))
        if ROMs[comp].scaler_red:
            #scaling all outputs
            output_data_predict_test = ROMs[comp].scaler_red.inverse_transform(output_data_predict_test)
        

        print("Test error")
        error_test = np.mean(output_data_test-output_data_predict_test)
        print(error_test)
        plt.plot(output_data_test[:,:10],"-")
        plt.plot(output_data_predict_test[:,:10],"--")
        plt.show()
        

        



if predict_phase:
    for comp in components:
        input_data_predict = np.load(input_test_file+".npy")
        ROMs[comp] = ROM.load(folder+"/ezyrb_"+comp+".rom")

        output_data_predict = ROMs[comp].approximation.predict(input_data_predict)
        if ROMs[comp].scaler_red:
            print("rescaling outputs")
            #scaling all outputs
            output_data_predict = ROMs[comp].scaler_red.inverse_transform(output_data_predict)


        np.save(output_test_file+"_"+comp+".npy", output_data_predict)


if False:
    for i in range(10):
        for comp in components:
            simul_folder = folder+"/param_%03d"%i
            input_test_file = simul_folder+"/training_input"
            input_data_predict = np.load(input_test_file+".npy")
            ROMs[comp] = ROM.load(folder+"/ezyrb_"+comp+".rom")
            output_data_predict = ROMs[comp].approximation.predict(input_data_predict)

            if ROMs[comp].scaler_red:
                print("rescaling outputs")
                #scaling all outputs
                output_data_predict = ROMs[comp].scaler_red.inverse_transform(output_data_predict)


            output_test_file = simul_folder+"/test_output"
            np.save(output_test_file+"_"+comp+".npy", output_data_predict)



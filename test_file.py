from scipy.io import loadmat
from models import interpolation, SRCNN_predict, DNCNN_predict
import numpy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    # load datasets
    channel_model = "VehA"
    SNR = 22
    Number_of_pilots = 48

    perfect = loadmat("Perfect_H_40000" + ".mat")['My_perfect_H']

    noisy_input = loadmat("My_noisy_H_" + str(SNR) + ".mat")['My_noisy_H']

    interp_noisy = interpolation(noisy_input, SNR, Number_of_pilots, 'rbf')


    perfect_image = numpy.zeros((len(perfect),72,14,2))
    perfect_image[:,:,:,0] = numpy.real(perfect)
    perfect_image[:,:,:,1] = numpy.imag(perfect)
    perfect_image = numpy.concatenate((perfect_image[:,:,:,0], perfect_image[:,:,:,1]), axis=0).reshape(2*len(perfect), 72, 14, 1)

    idx_random = numpy.random.rand(len(perfect_image))                                                                          < (1/9)  # uses 32000 from 36000 as training and the rest as validation
    train_data, train_label = interp_noisy[~idx_random,:,:,:] , perfect_image[~idx_random,:,:,:]
    val_data, val_label = interp_noisy[idx_random,:,:,:] , perfect_image[idx_random,:,:,:]


    path1 = 'result/03-23-21-29/SRCNN/'
    path2 = 'result/03-23-21-29/DNCNN/'


    srcnn_pred_test = SRCNN_predict(val_data, channel_model, Number_of_pilots, SNR, path1)
    dncnn_pred_test = DNCNN_predict(srcnn_pred_test, channel_model, Number_of_pilots, SNR, path2)

    aaaa = dncnn_pred_test

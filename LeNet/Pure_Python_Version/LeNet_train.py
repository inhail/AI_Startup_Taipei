from numpy import *
from ConvNet import *
import time
import struct
import os

#mnist has a training set of 60,000 examples, and a test set of 10,000 examples.
#log檔作用:紀錄檔案（logfile）是一個記錄了發生在執行中的作業系統或其他軟體中的事件的檔案
#
def train_net(train_covnet, logfile, cycle, learn_rate, case_num = -1) :
    # Read data 
    # Change it to your own dataset path
    trainim_filepath = './data/raw/train-images.idx3-ubyte' #training的資料
    trainlabel_filepath = './data/raw/train-labels.idx1-ubyte' #label的資料
    
    trainimfile = open(trainim_filepath, 'rb') #open()開啟檔案
    trainlabelfile = open(trainlabel_filepath, 'rb') #使用’rb’按照二進位制位進行讀取的，不會將讀取的位元組轉換成字元 
   
    train_im = trainimfile.read() # 讀取文件內容 f.read(size) - 回傳檔案內容,
    train_label = trainlabelfile.read() #size為要讀取進來的字串長度，若不填則讀取整份文件
    im_index = 0
    label_index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , train_im , im_index)
    magic, numLabels = struct.unpack_from('>II', train_label, label_index)
    print ('train_set:', numImages)

    train_btime = time.time()
    logfile.write('learn_rate:' + str(learn_rate) + '\t')
    logfile.write('train_cycle:' + str(cycle) + '\t')
########################################################################################################
    # Begin to train
    for c in range(cycle) :
        im_index = struct.calcsize('>IIII')
        label_index = struct.calcsize('>II')
        train_case_num = numImages
        if case_num != -1 and case_num < numImages :
            train_case_num = case_num
        logfile.write("trainset_num:" + str(train_case_num) + '\t')
        for case in range(train_case_num) :
            im = struct.unpack_from('>784B', train_im, im_index)
            label = struct.unpack_from('>1B', train_label, label_index)
            im_index += struct.calcsize('>784B')
            label_index += struct.calcsize('>1B')
            im = array(im)
            im = im.reshape(28,28)
            bigim = list(ones((32, 32)) * -0.1)
            for i in range(28) :
                for j in range(28) :
                    if im[i][j] > 0 :
                        bigim[i+2][j+2] = 1.175
            im = array([bigim])
            label = label[0]
            print (case, label)
            train_covnet.fw_prop(im, label)
            train_covnet.bw_prop(im, label, learn_rate[c])

    print ('train_time:', time.time() - train_btime)
    logfile.write('train_time:'+ str(time.time() - train_btime) + '\t')
######################################################################################################
def test_net(train_covnet, logfile, case_num = -1) :
    
    # Read data 
    # Change it to your own dataset path
    testim_filepath = './data/raw/t10k-images.idx3-ubyte'
    testlabel_filepath = './data/raw/t10k-labels.idx1-ubyte'
    testimfile = open(testim_filepath, 'rb')
    testlabelfile = open(testlabel_filepath, 'rb')
    test_im = testimfile.read()
    test_label = testlabelfile.read()

    im_index = 0
    label_index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , test_im , im_index)
    magic, numLabels = struct.unpack_from('>II', test_label, label_index)
    print('test_set:', numImages)
    im_index += struct.calcsize('>IIII')
    label_index += struct.calcsize('>II')
    
    correct_num = 0
    testcase_num = numImages
    if case_num != -1 and case_num < numImages:
        testcase_num = case_num
    logfile.write("testset_num:" + str(testcase_num) + '\t')

    # To test
    for case in range(testcase_num) :
        im = struct.unpack_from('>784B', test_im, im_index)
        label = struct.unpack_from('>1B', test_label, label_index)
        im_index += struct.calcsize('>784B')
        label_index += struct.calcsize('>1B')
        im = array(im)
        im = im.reshape(28,28)
        bigim = list(ones((32, 32)) * -0.1)
        for i in range(28) :
            for j in range(28) :
                if im[i][j] > 0 :
                    bigim[i+2][j+2] = 1.175
        im = array([bigim])
        label = label[0]
        print( case, label)
        train_covnet.fw_prop(im)
        if argmax(train_covnet.outputlay7.maps[0][0]) == label :
            correct_num += 1
    correct_rate = correct_num / float(testcase_num)
    print('test_correct_rate:', correct_rate)
    logfile.write('test_correct_rate:'+ str(correct_rate) + '\t')
    logfile.write('\n')


log_timeflag = time.time()
train_covnet = CovNet()
# Creat a folder name 'log' to save the history
train_covnet.print_netweight('log/origin_weight' + str(log_timeflag) + '.log')
logfile = open('log/nanerrortestcase.log', 'w')
logfile.write("train_time:" + str(log_timeflag) + '\t')
train_net(train_covnet, logfile, 1, [0.0001, 0.0001], 50)
train_covnet.print_netweight('log/trained_weight' + str(log_timeflag) + '.log')
test_net(train_covnet, logfile, 50)
logfile.write('\n')
logfile.close()

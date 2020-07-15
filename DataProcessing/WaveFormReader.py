import numpy as np
import wfdb
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import MinMaxScaler
import os
from scipy import signal
classify_label=['A' ,'F' ,'N' ,'R' ,'V' ]

#https://arxiv.org/pdf/1805.00794 annotate
#call get_list_patience)data to get data
def pad(signal, size=240):
    res=[]
    if len(signal[0])>size:
        return signal[:,0:size]
    for lead in signal:
        res.append(np.concatenate((lead,np.array([0]*(size-len(lead))))))
    return np.array(res)


def zero_one_scale(patience_signal):
    print(patience_signal)
    res=[]
    for signal in patience_signal:
        s=MinMaxScaler().fit(signal.reshape(-1, 1)).transform(signal.reshape(-1, 1)).ravel()
        res.append(s)
    return np.asarray(res)
def get_single_patiennce_data(patience,dataset_path='Data/StPeterburg/',):
    PatienceData={}
    for label in classify_label:
        PatienceData[label]=[]
    sample_rate = wfdb.rdheader(dataset_path  + patience).__dict__['fs']
    all_middle_qrs_annotation = wfdb.rdann(dataset_path  + patience, 'atr').sample
    all_middle_qrs_label = wfdb.rdann(dataset_path + patience, 'atr').symbol
    patience_signal,_= wfdb.rdsamp(dataset_path + patience)
    patience_signal = np.asarray(patience_signal).T

    curent_signal_idx=0
    curent_annotation_idx=0

    while curent_signal_idx<len(patience_signal[0]):

        middle_qrs_of_signal=[]
        middle_qrs_label_of_signal=[]
        while  curent_annotation_idx<len(all_middle_qrs_annotation) and all_middle_qrs_annotation[curent_annotation_idx]<=curent_signal_idx+10*sample_rate :
            middle_qrs_of_signal.append(all_middle_qrs_annotation[curent_annotation_idx])
            middle_qrs_label_of_signal.append(all_middle_qrs_label[curent_annotation_idx])
            curent_annotation_idx+=1

        mean_interval=(middle_qrs_of_signal[-1]-middle_qrs_of_signal[0])//len(middle_qrs_of_signal)
        # print(middle_qrs_of_signal)
        # print(middle_qrs_label_of_signal)
        for idx,label in enumerate(middle_qrs_label_of_signal):
            if label in classify_label:
                PatienceData[label].append(pad(zero_one_scale(patience_signal[:,middle_qrs_of_signal[idx]:middle_qrs_of_signal[idx]+int(1.2*mean_interval)])))

        curent_signal_idx+=10*sample_rate
    return PatienceData


def get_list_patience_data(patience_list):
    Signal_data=[]
    Label_data=[]

    for patience in patience_list:
        # print(patience)
        Patience_Data=get_single_patiennce_data(patience)

        for label in Patience_Data:
            for Signal in Patience_Data[label]:
                Signal_data.append(Signal)
                Label_data.append(label)

    return (Signal_data,Label_data)

#https://arxiv.org/pdf/1812.07421v2.pdf annotate

def middleqrs_to_r(l2signal,midle_qrs_locations):
    offset=8
    r_peak_locations=[]
    for middle_qrs_peak in midle_qrs_locations:
        r_peak_locations.append(middle_qrs_peak-offset+np.argmax(l2signal[middle_qrs_peak-offset:middle_qrs_peak+offset]))
    return r_peak_locations


def onoffset(interval,mode):
    slope=[]
    for i in range(1,len(interval)-1):
        slope.append(interval[i+1]-interval[i-1])

    if mode=='on':
        return np.argmin(np.abs(slope))
    elif mode=='off':
        slope_thres=0.2*np.max(np.abs(slope))
        slope_s=np.where(np.abs(slope)>=slope_thres)
        return slope_s[0][0]

def t_wave_extractor(ecg_signals,r_positons,fs,annotation):
    aveHB=len(ecg_signals)/len(r_positons)
    refine_r_positions=[]
    refine_annotation=[]
    for idx,r_pos in enumerate(r_positons):
        if annotation[idx] not in ['~','+']:
            refine_r_positions.append(r_pos)
            refine_annotation.append(annotation[idx])

    r_positons=refine_r_positions
    annotation=refine_annotation
    fid_pks=np.zeros(shape=(len(r_positons),7))
    windowS = round(fs * 0.1)
    windowQ = round(fs * 0.05)
    windowP = round(aveHB / 3)
    windowT = round(aveHB * 2 / 3)
    windowOF = round(fs * 0.04)
    for i in range(len(r_positons)):
        this_r=r_positons[i]
        if i==0:
            fid_pks[i][3] = this_r
            fid_pks[i][5] = this_r + windowS
        elif  i == len(r_positons)-1:
            fid_pks[i][3] = this_r
            fid_pks[i][1] = this_r - windowQ

        else:
            if (this_r+windowT)<len(ecg_signals) and this_r-windowP>=0:
                fid_pks[i][3]=this_r
                s_index=np.argmin(ecg_signals[this_r:this_r+windowS])
                this_s=s_index+this_r
                fid_pks[i][4]=this_s
                q_index=np.argmin(ecg_signals[this_r-windowQ:this_r])
                this_q=this_r+q_index-windowQ
                fid_pks[i][2]=this_q
                interval_q=ecg_signals[this_q-windowOF:this_q]
                this_on=this_q-windowOF+onoffset(interval_q,'on')
                interval_s=ecg_signals[this_s:this_s+windowOF]
                this_off= this_s +onoffset(interval_s, 'off')
                fid_pks[i][1]=this_on
                fid_pks[i][5]=this_off
    for i in range(0,len(r_positons)-1):
        if fid_pks[i][3]==0:
            continue
        if i!=0:
            last_off=fid_pks[i-1][5]
            this_on=fid_pks[i][1]
        this_off=fid_pks[i][5]
        next_on=fid_pks[i+1][1]
        if this_off<next_on:
            candidate_thres=0.33
            Tzone=np.asarray([this_off,next_on-round(candidate_thres*(next_on-this_off))],dtype=np.int)
            fid_pks[i][6]=this_off+np.argmax(ecg_signals[Tzone[0]:Tzone[1]])
            if i!=0 and this_on>last_off +2:
                Pzone = np.asarray([last_off + round((1-candidate_thres)*(this_on - last_off) ), this_on], dtype=np.int)
                fid_pks[i][0] = (last_off + round((1-candidate_thres)*(this_on - last_off))) + np.argmax(ecg_signals[Pzone[0]:Pzone[1]])
    ecg_peaks=[]
    ecg_anno=[]
    for i in range(len(r_positons)):
        if fid_pks[i][6] !=0:
            ecg_peaks.append(fid_pks[i,:])
            ecg_anno.append(annotation[i])
    return np.asarray(ecg_peaks,dtype=np.int),ecg_anno

def normalize(x):
    min_value=-13.04
    max_value=11.55
    return x#(x-min_value)/(max_value-min_value)

def convert_to_mit_anno(label):
    if label in ['L','N','R','e','j']:
        return 'N'
    elif label in ['E','V']:
        return 'V'
    elif label=='F':
        return 'F'
    elif label in ['A','J','S','a']:
        return 'S'
    # elif label in ['|','Q','f']:
    #     return 'Q'
    else:
        return 'none'

def get_data(file_path,resample_rate=280,use_mit=True,list_patience=None,mode='train',test_mode='intra'):
    if use_mit and test_mode=='intra':
        if mode=='train':
            list_patience=[101, 106, 108,109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223,230]
        else:
            list_patience=[100]#[100, 103, 105,111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    else:
        list_patience=[]
        for patience in  os.listdir(file_path):
            if patience.endswith('hea'):
                list_patience.append(patience[:-4])
        if use_mit:
            remove_patience=[102,104,107,217]
        for patience in remove_patience:
            list_patience.remove(remove_patience)
    data_patience={}
    for patience in list_patience:
        print(patience)
        patience_signals,patience_info= wfdb.rdsamp(file_path  +'/'+str(patience))
        leadII_index=np.where(np.asarray(patience_info['sig_name']) == 'MLII')[0][0] if use_mit else\
              np.where(np.asarray(patience_info['sig_name']) == 'II')[0][0]
        patience_signal = np.asarray(patience_signals)[:,leadII_index]
        sample = wfdb.rdann(file_path  +'/'+ str(patience), 'atr').sample[1:]
        symbol = wfdb.rdann(file_path  + '/'+str(patience), 'atr').symbol[1:]
        fs = wfdb.rdheader(file_path + '/'+str(patience)).__dict__['fs']
        ecg_peaks,ecg_anno=t_wave_extractor(patience_signal,sample,fs,symbol)
        patience_ecg=[]
        patience_anno=[]
        for i in range(1,len(ecg_peaks)):
            if convert_to_mit_anno(ecg_anno[i]) !='none':
                normalize_signal=normalize(signal.resample(patience_signal[ecg_peaks[i-1][6]:ecg_peaks[i][6]],resample_rate))
                if not np.isnan(normalize_signal)[0]:
                    patience_ecg.append(normalize_signal)
                    patience_anno.append(convert_to_mit_anno(ecg_anno[i]))
            
        data_patience[patience]=[patience_ecg,patience_anno] 

        # print(len(patience_ecg))
        # print(len(patience_anno))
        # print('----------')
        # for i in range(len(data_patience[patience][0])):
        #     if data_patience[patience][1][i]!='C':
        #         plt.plot(data_patience[patience][0][i])
        #         plt.text(100,0.55,data_patience[patience][1][i])
        #         plt.show()
    return data_patience


if __name__=='__main__':
    np.set_printoptions(suppress=True)
    dataset_path='Data/mit-bih'
    data_patience=get_data(dataset_path,use_mit=True,mode='train')
    anno_stat={'N':0,'V':0,'F':0,'S':0,'Q':0}
    for patience in data_patience:
        for idx,anno in enumerate(data_patience[patience][1]):
            anno_stat[anno]+=1
            # if anno!='N' and anno!='V':
            #     plt.plot(data_patience[patience][0][idx])
            #     plt.text(100,1,data_patience[patience][1][idx])
            #     plt.show()
    print(anno_stat)

    # print(data_patience)

    # # patience_list = []
    # # for idx in range(75):
    # #     patience_list.append('I0' + str(idx + 1) if idx <= 8 else 'I' + str(idx + 1))



    patience_list= ['116']#101, 106, 108,109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223,230]
    for patience in patience_list:
        patience_signals,patience_info= wfdb.rdsamp(dataset_path  +'/'+str(patience))
        print(patience_info)
    for patience in patience_list:
        patience=str(patience)
        patience_signals,patience_info= wfdb.rdsamp(dataset_path  +'/'+ patience)
        leadII_index=np.where(np.asarray(patience_info['sig_name']) == 'MLII')[0][0]
        patience_signal = np.asarray(patience_signals)[:,0]
        sample = wfdb.rdann(dataset_path  +'/'+ str(patience), 'atr').sample[1:]
        #sample = middleqrs_to_r(np.asarray(patience_signals).T[0], sample) ###
        symbol = wfdb.rdann(dataset_path  + '/'+str(patience), 'atr').symbol[1:]
        print(symbol)
        view_start_index=1860
        view_end_index=1870
        print(symbol[view_start_index:view_end_index])
        offset=5
        test_signal=patience_signal[-offset+sample[view_start_index]:sample[view_end_index - 1] + offset]
        ecg_peaks,_ = t_wave_extractor(test_signal,sample[view_start_index:view_end_index]-sample[view_start_index]+offset,360,symbol[view_start_index:view_end_index])
        t_wave=np.asarray(ecg_peaks,dtype=np.int)[:,6]
        plt.plot(test_signal)
        plt.scatter(t_wave,test_signal[t_wave],facecolor='red',s=20)
        plt.scatter(sample[view_start_index:view_end_index]-sample[view_start_index]+offset,test_signal[sample[view_start_index:view_end_index]-sample[view_start_index]+offset],facecolor='green',s=20)
        plt.show()
'''
def convert_to_mit_anno(label):
    if label =='N':
        return 'N'
    elif label in['E','V']:
        return 'V'
    elif label in ['A','J','S','a']:
        return 'S'
    elif label=='L':
        return 'L'
    elif label =='R':
        return 'R'
    else:
        return 'none'
'''
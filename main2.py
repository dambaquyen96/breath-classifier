from __future__ import print_function
import argparse
import os
import essentia
import essentia.standard
import essentia.streaming
from essentia.standard import *
import librosa.feature
import numpy as np
import logging

logging.basicConfig(format='[%(levelname)s|%(asctime)s] %(message)s',
                    datefmt='%Y%m%d %H:%M:%S',
                    level=logging.DEBUG)

global ARGS

def create_weka_gfcc():
    """
    Create weka data file with feature: GFCC 40 coeffs
    """
    global ARGS

    name = '_'.join(ARGS.labels.split(','))

    fout = open('weka_new/GFCC80_{}.arff'.format(name), 'w')
    fout.write('@RELATION {}_dataset\n\n'.format(name))

    fout.write('@ATTRIBUTE MEAN_GFCC1	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC2	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC3	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC4	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC5	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC6	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC7	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC8	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC9	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC10	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC11	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC12	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC13	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC14	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC15	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC16	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC17	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC18	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC19	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC20	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC21	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC22	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC23	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC24	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC25	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC26	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC27	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC28	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC29	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC30	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC31	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC32	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC33	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC34	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC35	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC36	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC37	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC38	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC39	REAL\n')
    fout.write('@ATTRIBUTE MEAN_GFCC40	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC1	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC2	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC3	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC4	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC5	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC6	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC7	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC8	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC9	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC10	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC11	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC12	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC13	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC14	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC15	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC16	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC17	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC18	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC19	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC20	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC21	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC22	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC23	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC24	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC25	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC26	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC27	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC28	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC29	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC30	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC31	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC32	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC33	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC34	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC35	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC36	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC37	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC38	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC39	REAL\n')
    fout.write('@ATTRIBUTE STD_GFCC40	REAL\n')
    fout.write('@ATTRIBUTE class 	{'+ARGS.labels+'}\n\n')
    
    fout.write('@DATA\n')

    windowing = Windowing(type='hamming',
                        size=1104,
                        zeroPhase=False)
    spectrum = Spectrum(size=1104)
    gfcc = GFCC(highFrequencyBound=6000,
                inputSize=552,
                lowFrequencyBound=0,
                numberBands=40,
                numberCoefficients=40,
                sampleRate=44100)

    logging.info('Preprocess label <{}>'.format(ARGS.txt))
    list_label = ARGS.labels.split(',')
    sample_label = [4]*50020
    window_label = [4]*50000
    with open(ARGS.txt) as f:
        content = f.readlines()
        for line in content:
            tokens = line.split()
            start = int(float(tokens[0])*1000)
            end = int(float(tokens[1])*1000)
            si = start / 100
            ei = end / 100
            if ei % 100 != 0:
                ei += 1
            idx_label = list_label.index(tokens[2])
            if idx_label == 4:
                continue
            for i in range(si, ei+1):
                sample_label[i] = idx_label
    for i in range(0, len(window_label)):
        ok = 0
        for j in range(i, i+20):
            if sample_label[j] != 4:
                if ok == 0:
                    window_label[i] = sample_label[j]
                    ok = 1
                else:
                    if window_label[i] != sample_label[j]:
                        window_label[i] = -1
    # print(sample_label)               
    # print(window_label)           
    # print(list_label)  
    # return
    
    logging.info('Process <{}>'.format(ARGS.wav))
    loader = MonoLoader(filename=ARGS.wav, sampleRate=ARGS.sampleRate)
    audio = loader()
    idx = 0
    cnt = 0
    for window in FrameGenerator(audio, 
                                frameSize=ARGS.window_length*ARGS.sampleRate/1000, 
                                hopSize=ARGS.window_stride*ARGS.sampleRate/1000, 
                                startFromZero=True):
        if window_label[idx] == -1:
            logging.info('Index {} failed'.format(idx))
            idx += 1
            continue
        label = list_label[window_label[idx]]
        gfccs = []
        for frame in FrameGenerator(window, 
                                    frameSize=ARGS.frame_length*ARGS.sampleRate/1000, 
                                    hopSize=ARGS.frame_stride*ARGS.sampleRate/1000, 
                                    startFromZero=True):
            s = spectrum(windowing(frame))
            _, g = gfcc(s)
            gfccs.append(g)
        gfccs = np.array(gfccs)
        gfccs_mean = np.mean(gfccs, axis=0)
        gfccs_std = np.std(gfccs, axis=0)
        feat = np.concatenate((gfccs_mean, gfccs_std), axis=0).tolist()
        str_feat = [str(x) for x in feat]
        line = ','.join(str_feat)+','+label
        fout.write(line+'\n')
        cnt = cnt+1
        idx = idx+1
        if cnt % 1000 == 0:
            logging.info('Processed {} samples'.format(cnt))
    logging.info('Finish with {} samples'.format(cnt))

def create_weka_mfcc_13():
    """
    Create weka data file with feature: MFCC 13 coeffs, it's delta & double-delta (Total: 39)
    """
    global ARGS

    name = '_'.join(ARGS.labels.split(','))

    fout = open('weka/MFCC78_TUNNING_{}.arff'.format(name), 'w')
    fout.write('@RELATION {}_dataset\n\n'.format(name))

    fout.write('@ATTRIBUTE MEAN_MFCC1	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC2	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC3	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC4	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC5	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC6	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC7	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC8	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC9	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC10	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC11	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC12	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC13	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD1	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD2	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD3	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD4	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD5	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD6	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD7	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD8	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD9	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD10	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD11	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD12	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD13	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD1	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD2	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD3	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD4	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD5	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD6	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD7	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD8	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD9	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD10	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD11	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD12	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD13	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC1	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC2	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC3	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC4	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC5	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC6	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC7	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC8	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC9	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC10	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC11	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC12	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC13	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD1	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD2	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD3	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD4	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD5	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD6	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD7	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD8	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD9	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD10	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD11	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD12	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD13	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD1	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD2	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD3	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD4	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD5	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD6	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD7	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD8	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD9	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD10	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD11	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD12	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD13	REAL\n')
    fout.write('@ATTRIBUTE class 	{'+ARGS.labels+'}\n\n')
    
    fout.write('@DATA\n')

    windowing = Windowing(type='hamming')
    spectrum = Spectrum()
    mfcc = MFCC(highFrequencyBound=6000,
                inputSize=552,
                lowFrequencyBound=0,
                numberBands=40,
                numberCoefficients=13,
                sampleRate=44100)
    for label in ARGS.labels.split(','):
        dir = os.path.join(ARGS.dir, label)
        logging.info('Access folder <{}>'.format(dir))
        for file in sorted(os.listdir(dir)):
            if file.endswith('.wav'):
                logging.info('Process <{}>'.format(file))
                path = os.path.join(dir, file)
                loader = MonoLoader(filename=path, sampleRate=ARGS.sampleRate)
                audio = loader()
                cnt = 0
                for window in FrameGenerator(audio, 
                                            frameSize=ARGS.window_length*ARGS.sampleRate/1000, 
                                            hopSize=ARGS.window_stride*ARGS.sampleRate/1000, 
                                            startFromZero=True):
                    mfccs = []
                    for frame in FrameGenerator(window, 
                                                frameSize=ARGS.frame_length*ARGS.sampleRate/1000, 
                                                hopSize=ARGS.frame_stride*ARGS.sampleRate/1000, 
                                                startFromZero=True):
                        s = spectrum(windowing(frame))
                        _, m = mfcc(s)
                        m_delta = librosa.feature.delta(m, order=1)
                        m_delta_delta = librosa.feature.delta(m, order=2)
                        m_all = np.concatenate((m, m_delta, m_delta_delta), axis=0)
                        mfccs.append(m_all)
                    mfccs = np.array(mfccs)
                    mfccs_mean = np.mean(mfccs, axis=0)
                    mfccs_std = np.std(mfccs, axis=0)
                    feat = np.concatenate((mfccs_mean, mfccs_std), axis=0).tolist()
                    str_feat = [str(x) for x in feat]
                    line = ','.join(str_feat)+','+label
                    fout.write(line+'\n')
                    cnt = cnt+1
                logging.info('{} samples'.format(cnt))

def main():
    global ARGS
    parser = argparse.ArgumentParser(description='Parser arguments')
    parser.add_argument(
        '--dir',
        type=str, 
        default='dataset2',
        help='Root dir of dataset')
    parser.add_argument(
        '--wav',
        type=str, 
        default='aaa',
        help='wave file')
    parser.add_argument(
        '--txt',
        type=str, 
        default='txt',
        help='Label text file')
    parser.add_argument(
        '--labels',
        type=str,
        default='slight_breath,normal_breath,deep_breath,strong_breath,unknown',
        help='Label name of each class')
    parser.add_argument(
        '--sampleRate',
        type=int,
        default=44100,
        help='Sample rate of audio')
    parser.add_argument(    
        '--window_length',
        type=int,
        default=2000,
        help='Window length of each sample in ms')
    parser.add_argument(
        '--window_stride',
        type=int,
        default=100,
        help='Window stride of each sample in ms')
    parser.add_argument(
        '--frame_length',
        type=int,
        default=25,
        help='Frame length of each sample in ms')
    parser.add_argument(
        '--frame_stride',
        type=int,
        default=10,
        help='Frame stride of each sample in ms')
    parser.add_argument(
        '--feature',
        type=str,
        choices=['mfcc', 'gfcc'],
        default='mfcc',
        help='Type of features')
    ARGS = parser.parse_args()
    if ARGS.feature == 'mfcc':
        create_weka_mfcc_13()
    elif ARGS.feature == 'gfcc':
        create_weka_gfcc()

if __name__ == "__main__":
    main()
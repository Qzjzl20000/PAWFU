import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np

# Label in MELD {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}


class MELDDataset(Dataset):

    def __init__(self, train=True, if_raw_data=False):

        if if_raw_data:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, self.testVid, _ = pickle.load(
                open('Datasets/MELD/MELD_raw.pkl', 'rb'), encoding='latin1')
            # # EmoBERTa
            # self.videoText = pickle.load(
            #     open('Datasets/MELD/TextFeatures.pkl', 'rb'))
            # # OpenSMILE
            # self.videoAudio = pickle.load(
            #     open('Datasets/MELD/AudioFeatures.pkl', 'rb'))
            # # VisExtNet
            # self.videoVisual = pickle.load(
            #     open('Datasets/MELD/VisualFeatures.pkl', 'rb'))

        else:
            # raw data: self.videoText, self.videoAudio, self.videoVisual,
            self.videoIDs, self.videoSpeakers, self.videoLabels, _, _, _, self.videoSentence, self.trainVid, self.testVid, _ = pickle.load(
                open('Datasets/MELD/MELD_raw.pkl', 'rb'), encoding='latin1')
            # EmoBERTa
            self.videoText = pickle.load(
                open('Datasets/MELD/TextFeatures.pkl', 'rb'))
            # OpenSMILE
            self.videoAudio = pickle.load(
                open('Datasets/MELD/AudioFeatures.pkl', 'rb'))
            # VisExtNet
            self.videoVisual = pickle.load(
                open('Datasets/MELD/VisualFeatures.pkl', 'rb'))
            # # DenseNet
            # self.videoVisual = pickle.load(
            #     open('Datasets/MELD/Dense_VisualFeatures.pkl', 'rb'))

        # print("TAV shape:",self.videoText.size(),self.videoAudio.size(),self.videoVisual.size())
        # 检查 TAV 数据的维度
        if self.videoText and self.videoAudio and self.videoVisual:
            # 选择一个样本进行检查
            sample_id = list(self.videoText.keys())[0]
            text_feature = torch.tensor(self.videoText[sample_id])
            audio_feature = torch.tensor(self.videoAudio[sample_id])
            visual_feature = torch.tensor(self.videoVisual[sample_id])

            print("TAV shape:")
            print("Text feature shape:", text_feature.size())
            print("Audio feature shape:", audio_feature.size())
            print("Visual feature shape:", visual_feature.size())
        else:
            print("One or more of the TAV data dictionaries is empty.")

        self.trainVid = sorted(self.trainVid)
        self.testVid = sorted(self.testVid)

        self.samples = [
            sample for sample in (self.trainVid if train else self.testVid)
        ]
        self.len = len(self.samples)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        vid = self.samples[index]
        TextFeature = torch.FloatTensor(np.array(self.videoText[vid]))
        AudioFeature = torch.FloatTensor(np.array(self.videoAudio[vid]))
        VisualFeature = torch.FloatTensor(np.array(self.videoVisual[vid]))
        SpeakersMasks = torch.FloatTensor(np.array(
            self.videoSpeakers[vid]))  # one-hots (original data)
        UtteranceMasks = torch.FloatTensor(
            np.array([1] * len(self.videoLabels[vid])))
        GroundTruth = torch.LongTensor(np.array(self.videoLabels[vid]))

        return TextFeature, AudioFeature, VisualFeature, SpeakersMasks, UtteranceMasks, GroundTruth

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        output = []
        for i in dat:
            temp = dat[i].values
            if i <= 2:
                output.append(
                    pad_sequence([temp[i] for i in range(len(temp))],
                                 padding_value=0))
            elif i <= 4:
                output.append(
                    pad_sequence([temp[i] for i in range(len(temp))],
                                 True,
                                 padding_value=0))
            elif i <= 5:
                output.append(
                    pad_sequence([temp[i] for i in range(len(temp))],
                                 True,
                                 padding_value=-1))

        return output

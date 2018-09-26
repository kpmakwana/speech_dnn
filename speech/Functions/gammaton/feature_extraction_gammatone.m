clc; clear all; close all;

 for i=1:1:10
    myFile = sprintf('/Users/mihir/Documents/DA-IICT/Semester 6/CT478 - Speech Tech./Project/Speech_Database/noisy_trainset_wav_16k\');
    [y,fs] = audioread('/Users/mihir/Documents/DA-IICT/Semester 6/CT478 - Speech Tech./Project/Speech_Database/noisy_trainset_wav_16k.zip');

%% initialization

signal = y{i};
window_length = 160; % 10ms (beacuse sampling frequency is 16000 Hz)
window_shift = 80; %5ms (window_shift < window_length)
pre_emphesize_factor = 0.9;

%% main code

addpath('Users/mihir/projects/speech/Functions');
[impulse_response, filtered_signal, energy] = my_gammatone(signal, fs, window_length, window_shift, pre_emphesize_factor);
 end
 save(feature,filtered_signal);
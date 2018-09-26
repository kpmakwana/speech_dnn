clc;close all;clear;

window_length = 160; % 10ms (beacuse sampling frequency is 16000 Hz)
window_shift = 80; %5ms (window_shift < window_length)
pre_emphesize_factor = 0.9;

path1 = '/Users/student/Desktop/project/';
a = dir([path1,'*.wav']);

%path2 = '/Users/student/Desktop/project/clean_trainset_wav_16k/';
%b = dir([path2,'*.wav']);

savepath = '/Users/student/Desktop/project/';


for i = 1:length(a)
    [y1,fs1] = audioread([path1,a(i).name]);
    %[y2,fs2] = audioread([path2,b(i).name]);
    [~, filtered_signal1, energy1] = my_gammatone(y1, fs1, window_length, window_shift, pre_emphesize_factor);
    save([savepath,'noise_',num2str(i)]);
    %[~, filtered_signal2, energy2] = my_gammatone(y2, fs2, window_length, window_shift, pre_emphesize_factor);
    %if i==3001
        %filtered_noise = energy1;
        %filtered_clean = energy2;
    %else
        %filtered_noise =[filtered_noise,energy1];
        %filtered_clean =[filtered_clean,energy2];
    %end
end

%for j = 1 : length(filtered_noise)/1000
   %noisy = filtered_noise((1:64),((1000*(j-1))+1 : 1000*j)); 
   %clean = filtered_clean((1:64),((1000*(j-1))+1 : 1000*j));
   
   %save([savepath,'Batch_',num2str(j-1+1758)],'noisy','clean');
  %end
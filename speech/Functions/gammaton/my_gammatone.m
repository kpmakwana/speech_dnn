function [impulse_response, filtered_signal, energy] = my_gammatone(signal, fs, window_length, window_shift, pre_emphesize_factor)

%% Initialization

 flow = 50;
 fc = 1500;
 fhigh = fs/2;
 filter_per_ERB = 2.05;   % Changing for no. of filter 
 
 %% main code
 
 addpath(genpath('Users/student/Desktop/project/speech/Functions'));
 windowed_signal = hamming(window_length);
 B = [1 -pre_emphesize_factor];
 signal = filter(B,1,signal);
 
 analyzer = Gfb_Analyzer_new(fs, flow, fc, fhigh, filter_per_ERB);
 impulse = [1 zeros(1,799)];
 [impulse_response, analyzer] = Gfb_Analyzer_process(analyzer,impulse);
 impulse_response = real(impulse_response);
 
 filtered_signal = Gammatone_filter(signal, impulse_response);
 energy = energies(filtered_signal, windowed_signal, window_shift);

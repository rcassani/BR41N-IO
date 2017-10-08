% Using the e-puck device 
clear all;
close all;

% Adds the epuck functions to the MATLAB Path
folder = [pwd '\'];
addpath(genpath(pwd));

% Connect with the ePuck
epuck=ePicKernel;
epuck=connect(epuck,'COM22');

% Create a TCP/IP server, which receives the instructions for the ePuck
epuck_server = tcpip('0.0.0.0', 50000, 'NetworkRole', 'server');
epuck_server.InputBufferSize = 5000;
epuck_server.Timeout = 600;
disp('Waiting for connection of the client')

fopen(epuck_server);
disp('The e-puck is ready to receive orders: ')

while true
    %This ensures that we are reading only the newest commands
    bd = epuck_server.BytesAvailable;
    if bd > 3
        fread(epuck_server, epuck_server.BytesAvailable-3);    
    end
       
    left_pwr  = fread(epuck_server,1);
    right_pwr = fread(epuck_server,1);
    time_sec = fread(epuck_server,1);
    
    if ~any([left_pwr, right_pwr, time_sec])
        break
    end
    
    disp([left_pwr, right_pwr, time_sec])
    
    epuck=set(epuck,'speed',[left_pwr ,right_pwr]); 
    epuck=update(epuck);
    
    pause(time_sec/10)
    
    epuck=set(epuck,'speed',[0 ,0]);   %r and L recieved from python 
    epuck=update(epuck);
        
end 

% Disconnect
epuck=disconnect(epuck);


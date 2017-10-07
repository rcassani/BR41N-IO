# -*- coding: utf-8 -*-
'''
Example of using Myo client 

'''

from clientmyo import ClientMyo
import time


if __name__ == "__main__":
    
    # Connection paramters    
    myo_ip = '127.0.0.1'
    myo_port = 40000
    
    # INRS Myo MAC
    # Old F1:13:CA:63:69:76
    # New ED:A2:9A:4A:41:EE
    
    
#TODO automate the execution for the myoclient_vi.exe
    
    # Creates the Myo client and stablish connection with the server
    myo_client = ClientMyo(myo_ip, myo_port) 
    time.sleep(2)

    # Send vibration 
    myo_client.vibrate('short')
    # Pause 2 secoonds
    time.sleep(2)
    
    # Send vibration 
    myo_client.vibrate('medium')
    # Pause 2 secoonds
    time.sleep(2)

    # Send vibration 
    myo_client.vibrate('long')
    # Pause 2 secoonds
    time.sleep(2)

    # Disconnects from Server
    myo_client.disconnect()
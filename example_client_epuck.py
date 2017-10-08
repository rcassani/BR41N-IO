# -*- coding: utf-8 -*-
'''
Example of using e-puck client 

'''

from clientepuck import ClientEpuck
import time


if __name__ == "__main__":
    
    # Connection paramters    
    epuck_ip = '127.0.0.1'
    epuck_port = 50000
    
    # PORT in MSI is COM22 
       
    # Creates the e-puck client and stablish connection with the server
    epuck_client = ClientEpuck(epuck_ip, epuck_port) 
    time.sleep(2)

    # Set turn parameters for epuck turns
    epuck_client.set_turn_settings(200, 50, 1)

    # Turn Left
    epuck_client.send_instruction('left')
    # Pause 2 secoonds
    time.sleep(2)
    
    # Turn Right 
    epuck_client.send_instruction('right')
    # Pause 2 secoonds
    time.sleep(2)

    # Send Stop 
    epuck_client.send_instruction('stop')
    # Pause 2 secoonds
    time.sleep(2)

    # Disconnects from Server
    epuck_client.disconnect()
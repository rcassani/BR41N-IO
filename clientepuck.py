# -*- coding: utf-8 -*-

import socket

class ClientEpuck:   
    '''
    This class is to wrap of the TCP Client for the
    e-puck vibrations
    '''

    def __init__(self, ip, port):
        """
        Constructor method. This method create a TCP/IP client 
        
        Arguments:
            ip: the IP adress to be used to connect to the epuck Server.
            port: the TCP IP port.
        """
        print('new')

        self.ip = ip
        self.port = port
        
        # TCP/IP connection
        self.connect()
        

    def connect(self):
        """
        Connects to the e-puck TCP/IP Server
        reconnection attempt is unsuccessful.
        """
        print('Attempting connection')
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((self.ip, self.port))
            print('Connection successful')
        except:
            self.client = None
            print('Connection attempt unsuccessful')
            raise

    def disconnect(self):
        """
        Closes TCP IP
        """
        self.client.close()
        self.client = None
        print('Connection closed successfully')

    def send_instruction(self, action):
        '''
        Sends a command list of instructions to turn e-pcuk server
        command_list = [left_pwr, right_power, time_secs]
        With all values from 0 to 255
        '''
        print(action)
        if action == 'left':
            self.sendcommand(self.low_pwr)
            self.sendcommand(self.high_pwr)
            self.sendcommand(self.time_sec)
            
        elif action == 'right':
            self.sendcommand(self.high_pwr)
            self.sendcommand(self.low_pwr)
            self.sendcommand(self.time_sec)
            
        elif action == 'stop':
            self.sendcommand(0)
            self.sendcommand(0)
            self.sendcommand(0)
                        
    def end_server(self):
        '''
        Send the termination command to the Server
        '''
        self.sendcommand('c')
        self.disconnect()
        
    def sendcommand(self, command):
        """
        Sends an arbitrary char (UINT8)

        Arguments:
            command: byte to send
        """
        self.client.send(bytearray(chr(command),'ISO-8859-1'))
        
    def set_turn_settings (self, high_pwr = 200, low_pwr = 0, time_sec = 1):
        '''
        Sets the power for the motors, the values are in the range [0-255]
        '''
        self.high_pwr = high_pwr
        self.low_pwr = low_pwr
        self.time_sec = time_sec
        
        
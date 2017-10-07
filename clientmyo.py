# -*- coding: utf-8 -*-

import socket

class ClientMyo:   
    '''
    This class is to wrap of the TCP Client for the
    Myo vibrations
    '''

    def __init__(self, ip, port):
        """
        Constructor method. This method create a TCP/IP client 
        
        Arguments:
            ip: the IP adress to be used to connect to the Myo Server.
            port: the TCP IP port.
        """
        self.ip = ip
        self.port = port
        
        # TCP/IP connection
        self.connect()

    def connect(self):
        """
        Connects to the Myo TCP/IP Server
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

    def vibrate(self, vibration_type='short'):
        '''
        Sends the vibration command to the Myo server
        
        vibration type = 'short, medium or long'
                            <1 ,    ~1,   ~2 seconds
        '''
        if vibration_type == 'short':
            self.sendcommand('s')
        elif vibration_type == 'medium':
            self.sendcommand('m')
        elif vibration_type == 'long':
            self.sendcommand('l')
    
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
            command: char to send
        """
        self.client.send(bytearray(command,'ISO-8859-1'))
        
import os
import pprint
import pygame

class PS4Controller(object):
    """Class representing the PS4 controller. Pretty straightforward functionality."""

    controller = None
    axis_data = None
    button_data = None
    hat_data = None


# Test script for using pygame to read in ds4 controller
# with ds4drv  running as a daemon
# DS4  controller axis maps:
# Axis0: Left stick l-r (-1 left, 1 right)
# Axis1: Left stick u-d (-1 up, 1 down)
# Axis2: Left trigger (-1 unpressed, 1 completely pressed)
# Axis3: Right stick l-r (-1 left, 1 right)
# Axis4: Right stick u-d (-1 up, 1 down)
# Axis5: Right trigger (-1 unpressed, 1 completely pressed)

LEFT_ANALOG_LEFT_RIGHT = 0
LEFT_ANALOG_UP_DOWN = 1
RIGHT_ANALOG_LEFT_RIGHT = 3
RIGHT_ANALOG_UP_DOWN = 4

    def init(self):
        """Initialize the joystick components"""
        
        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

    def listen(self):
        """Listen for events to happen"""
        
        if not self.axis_data:
            self.axis_data = {}

        if not self.button_data:
            self.button_data = {}
            for i in range(self.controller.get_numbuttons()):
                self.button_data[i] = False

        if not self.hat_data:
            self.hat_data = {}
            for i in range(self.controller.get_numhats()):
                self.hat_data[i] = (0, 0)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    self.axis_data[event.axis] = round(event.value,2)
                elif event.type == pygame.JOYBUTTONDOWN:
                    self.button_data[event.button] = True
                elif event.type == pygame.JOYBUTTONUP:
                    self.button_data[event.button] = False
                elif event.type == pygame.JOYHATMOTION:
                    self.hat_data[event.hat] = event.value

                # Insert your code on what you would like to happen for each event here!
                # In the current setup, I have the state simply printing out to the screen.
                
                os.system('clear')
                #pprint.pprint(self.button_data)
                pprint.pprint(self.axis_data)
                #pprint.pprint(self.hat_data)

    def yaw():
        return axis_data[LEFT_ANALOG_LEFT_RIGHT] or 0
    def pitch():
        return axis_data[RIGHT_ANALOG_UP_DOWN] or 0
    def throttle():
        return axis_data[LEFT_ANALOG_UP_DOWN] or 0
    def roll()
        return axis_data[RIGHT_ANALOG_LEFT_RIGHT] or 0


if __name__ == "__main__":
    ps4 = PS4Controller()
    ps4.init()
    ps4.listen()
import pygame
import sys

# Initialize Pygame & Detect Joystick
pygame.init()
pygame.joystick.init()

joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
for joystick in joysticks:
	print("detected" + joystick.get_name())

# Main loop to check joystick commands
while True:
	for event in pygame.event.get():
			if event.type == pygame.QUIT:
				print("pygame.QUIT")
				pygame.quit()
				sys.exit()
			elif event.type == pygame.JOYAXISMOTION:
				print("pygame.JOYAXISMOTION")
				axis = event.axis
				value = joystick.get_axis(axis)
				print(f"Axis {axis} value: {value:.3f}")
			elif event.type == pygame.JOYBUTTONDOWN:
				print("pygame.JOYBUTTONDOWN")
				button = event.button
				print(f"Button {button} released")
			elif event.type == pygame.JOYBUTTONUP:
				print("pygame.JOYBUTTONUP")
				button = event.button
				print(f"Button {button} released")
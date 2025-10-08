import pygame
import time

pygame.mixer.init()
pygame.mixer.music.load("maja_aa_rha.mp3")
pygame.mixer.music.play(-1)  # loop indefinitely

# Keep script alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pygame.mixer.music.stop()

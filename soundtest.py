import pygame

pygame.init()
pygame.mixer.init()

try:
    pygame.mixer.music.load("C:/Users/anany/OneDrive/Documents/DROWSINESS_DETECTION/face_deface_har/face_detection/alaarum.wav")
    pygame.mixer.music.play()
    input("Press Enter to stop the sound...")
    pygame.mixer.music.stop()
except pygame.error as e:
    print("Error playing sound:", e)

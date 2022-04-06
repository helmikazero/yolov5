import simpleaudio as sa




def PLAY_WARNING():
    filename = 'audio\warning.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    # play_obj.wait_done()


# from playsound import playsound

# playsound('warning.wav')
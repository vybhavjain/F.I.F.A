import speech_recognition as sr
import sounddevice as sd
import wavio

def listen():
    
    
    r = sr.Recognizer()  #Class to recognize the speech

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Set minimum energy threshold to {}".format(r.energy_threshold))
        print('Say Something')
        audio=r.listen(source)
        
    try:
        print('Google thinks you said:\n' + r.recognize_google(audio))     #can be replaced by multiple other API's like bing, web speech,etc
    except:
        pass
    
    return r.recognize_google(audio)

def listen2(filename):
    fs = 44100  # Sample rate
    seconds = 8  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    
    wavio.write(filename+'.wav', myrecording, fs ,sampwidth=2)
    
    r = sr.Recognizer()
    with sr.WavFile(filename + ".wav") as source:              # use "test.wav" as the audio source
        audio = r.record(source)                        # extract audio data from the file
    print('Google thinks you said:\n' + r.recognize_google(audio))
    return r.recognize_google(audio)

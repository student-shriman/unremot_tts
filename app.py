# Getting Accelerator  ..
try:
  import os
  import torch
  import soundfile as sf
  print('Basic modules has been imported')
  
except:
  print('There are some issues with the import, Please install the basic modules and try again ..')

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# Load FastPitch
try:
  from nemo.collections.tts.models import FastPitchModel
  spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
  print('FastPitch model has been imported successfully  ..')
except:
  print('There is some problem with importing the model ..')

# Load vocoder
try:
  from nemo.collections.tts.models import HifiGanModel
  model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")
  print('Vocoder has been imported too ..')
except:
  print('Vocoder is not getting imported  ..')
  print('Congratts!! All modules has been imported successfully  ..')

##  Getting the Text inputs for TTS  ..
def get_input(input_mode):
  try:
    # Moving with Sample Text
    if input_mode=='sample text':
      input_text = "Text-speak is especially popular among the digital natives or net-generation."
      print(input_text)
      print('Please wait we are processing your inputs ..')

    elif input_mode=='custom text':
      input_text = input('Please type your words .. ')
      print(' Please wait we are processing your inputs ..')

    elif input_mode=='file':
      print('Okk Great! You need to provide a txt file here. I can only process a txt file')
      file_path = input('Please give the path of file to be processed .. ')
      file1 = open(file_path, "r" , encoding="utf8")
      file = file1.readlines()
      print(file) 
      file1.close()
      print('Okk great .. Please hold on, we are processing your text.')
  except:
    print('Please provide correct file path ..')
  return input_text

# TTS Synthesis with model ..
def get_speech(text):
  try:
    parsed = spec_generator.parse(text)
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    audio = model.convert_spectrogram_to_audio(spec=spectrogram)
    print('Congratts Your inputted Text has been processed ..')

    # Save the audio to disk in a file called speech.wav
    sf.write("speech.wav", audio.to('cpu').detach().numpy()[0], 22050)
    print ('Text-to-Speech Synthesis has been done and output file has been saved successfully ..')
    print('Please check "speech.wav" file .')
  except:
        print('There are some problems with TTS ..')

input_mode = input(' Please choose an option - "sample text, custom text, file" ')
if input_mode=="sample text" or "custom text" or "file":
  text = get_input(input_mode)
  get_speech(text)
  
else:
  print('There is something wrong with your selection. Please choose a correct option')
# %% Libs and Modules

from pathlib import Path as pth # files managment
import pandas as pd # data managment
from scipy.io import wavfile as wv # audio file read/write operations
import numpy as np # data processing
import crepe as cp # pitch tracking
import psola # time-frequency stretch-compress
import sounddevice as sd # play audio
from matplotlib import pyplot as plt # plot graphic

# %% Constants

VOICEBANK_NAME = "TestBank"
PROGRAM_PATH = pth.cwd()
VOICEBANK_PATH = PROGRAM_PATH.joinpath(VOICEBANK_NAME)

SR = 48000 # Sampling Rate
BASEBPM = 90 # Surrogate BPM

PITCH_REF = 440 # A4, in Hz
VB_PITCH_KEY = -21# C3, in Hz, recorded pitch reference
PITCH_GAP = { 'C': -9, 'D': -7, 'E': -5, 'F': -4, 'G': -2, 'A': 0, 'B': 2, }

# %% Pre-Loading the Otoing Table

oto_path = VOICEBANK_PATH.joinpath('oto.ini')

global df_oto

df_oto = pd.read_csv(oto_path, header = None, 
                        names = ['Filename=Alias', 'Offset', 'Consonant',
                         'Cutoff', 'Preuterance', 'Overlap'], 
                        dtype =
                        {'Filename=Alias': str, 
                         'Offset': np.int32, 'Consonant': np.int32,
                         'Cutoff': np.int32, 'Preuterance': np.int32, 
                         'Overlap': np.int32})

# Processing Otoing Table

df_oto[['Filename', 'Alias']] = \
df_oto['Filename=Alias'].str.split('=', expand=True)

df_oto = df_oto.drop(columns = 'Filename=Alias')

# %% Reading Interpration File

INTER_FILE_NAME = "wave.txt"
inter_file_path = PROGRAM_PATH.joinpath(INTER_FILE_NAME)

with inter_file_path.open() as file:
    inter_header = file.readline()
    
tempo_info = inter_header.split(',')
bpm = float(tempo_info[0].split('=')[1])
str_fig = tempo_info[1].split('=')[1]
num, den = str_fig.split('/')
fig = float(num)/float(den)

global df_inter

df_inter = pd.read_csv(inter_file_path, header = 0, 
                        names = ['Alias', 'Pitch', 'Duration', 'Volume'], 
                        dtype =
                        {'Alias': str, 'Pitch': str, 
                         'Duration': str, 'Volume': np.float32 })

# Processing Interpration File

# %%Calculating the Real Pitch
 
pitch_name = df_inter['Pitch']
pitch_class = pitch_name.str[0].to_string(index = False).split()
pitch_scale = pitch_name.str[-1].to_numpy(dtype=np.float32)
up_shift = [(p.find('#') + 1)/2 for p in pitch_name] # sus
down_shift = [(p.find('b') + 1)/2 for p in pitch_name] # bemol
pitch_gaps = [PITCH_GAP[p] for p in pitch_class]

pitch_base_key = (pitch_scale - 4)*12 + pitch_gaps + up_shift + down_shift 
real_pitch = np.power(2,(np.float32(pitch_base_key)/12))*PITCH_REF # in Hz

df_inter['Real Pitch'] = pd.Series(real_pitch)

# %%Calculating the Real Duration

each_duration = df_inter['Duration'].to_string(index = False).split('\n')
rationals = [d.split('/') for d in each_duration]
num = np.array([i[0] for i in rationals])
den = np.array([i[1] for i in rationals])
float_duration = np.float32(num)/np.float32(den)

real_duration = ((60/bpm)/fig)*float_duration*1000 # in ms

df_inter['Real Duration'] = pd.Series(real_duration)


# %% Extract Acoustic Unit

def extractAcousticUnit(mask):
    info = df_oto.loc[df_oto['Alias'] == mask]
    FS, sample = wv.read( VOICEBANK_PATH.joinpath( 
        info['Filename'].values[0]))
    return(sample[np.int32(FS*np.single(info['Offset'].values[0])/1000) : \
                  -np.int32(FS*np.single(info['Cutoff'].values[0])/1000)])

# %% Extract Acoustic Unit Parameters
    # 0: Overlap
    # 1: Consonant
    # 2: Preuterance

def extractAcousticUnitParams(mask):
    info = df_oto.loc[df_oto['Alias'] == mask]
    return([info['Overlap'], info['Consonant'], info['Preuterance']])

# %% Generating Frequency Files

for order, note in df_inter.iterrows():
    frq_file_path = VOICEBANK_PATH.joinpath(note['Alias'] + '.frq')
    if not frq_file_path.is_file(): # avoid overwritting
        sample = extractAcousticUnit(note['Alias'])
        frq_sample = cp.predict(sample, SR)
        tracking = {'Timestamp': frq_sample[0], 'Pitch': frq_sample[1]}
        df_freq = pd.DataFrame(tracking)
        df_freq.to_csv(frq_file_path, index = False)

# %% Unit Duration/Pitch Modification

P_FLAT = False
P_PRESERVE = True

P_STM = 1

VIB_ON = False
VIB_HEIGHT = 0.01 # vibrato range
VIB_SPEED = 0.8 # vibrato speed

JIT_ON = False
JIT_SUP = 0.01
JIT_INF = 0.01
JIT_STEP = 0.005

def modifyUnitDurationandPitch(unit, pitch_contour, 
                               desired_duration, desired_pitch):
    time_stretch = (len(unit)/48)/desired_duration
    
    ref_pitch = np.power(2,(np.float32(VB_PITCH_KEY)/12))*PITCH_REF
    pitch_stretch = desired_pitch/ref_pitch
    
    if P_FLAT:
        flat_contour = desired_pitch*np.ones(shape = (len(pitch_contour),))
        return psola.vocode(unit, SR, constant_stretch = time_stretch,
                            target_pitch = flat_contour)
    elif P_PRESERVE:
        return psola.vocode(unit, SR, constant_stretch = time_stretch,
                            target_pitch = pitch_stretch*pitch_contour)
    
    if VIB_ON:
        vibrato = VIB_HEIGHT*np.sin(VIB_SPEED*np.arange(len(pitch_contour)))
        plt.subplot(4,1,1)
        plt.plot(vibrato)
    else:
        vibrato = np.ones(len(pitch_contour))
    
    if JIT_ON:
        jitter = np.zeros(shape=(len(pitch_contour, )))
        jitter[0] = JIT_STEP*(2*np.random.random_sample() - 1)
        for j in range(len(pitch_contour) - 1):
            step = JIT_STEP*(2*np.random.random_sample() - 1)
            if (jitter[j] + step > JIT_SUP) or ((jitter[j] + step < JIT_INF)):
                jitter[j + 1] = jitter[j] - step
            else:
                jitter[j + 1] = jitter[j] + step
        plt.subplot(4,1,2)
        plt.plot(jitter)
    else:
        jitter = np.ones(len(pitch_contour))
    variation = np.multiply((1 + vibrato),(1 + jitter))
    plt.subplot(4,1,3)
    plt.plot(variation)
    smooth_variation = np.convolve(variation, np.ones(P_STM), 'same') / P_STM
    plt.subplot(4,1,4)
    plt.plot(smooth_variation)
    
    smooth_pitch_contour = smooth_variation*desired_pitch
    
    return psola.vocode(unit, SR, constant_stretch = time_stretch,
                        target_pitch = smooth_pitch_contour)

# %% Unit Volume Modificatiom

SHIMMER = 0.001 # shimmer range

def modifyUnitVolume(unit, volume_contour):
    shimm = SHIMMER*(2*np.random.random(len(unit)) - 1)
    shiny_volume_contour = np.multiply((1 - shimm), volume_contour)
    return(np.multiply(shiny_volume_contour, unit))

# %% Music Interpretation

notes = [] # list of notes
overlaps = [] # list of overlap values

for order, note in df_inter.iterrows():
    unit = extractAcousticUnit(note['Alias'])
    
    frq_file_path = VOICEBANK_PATH.joinpath(note['Alias'] + '.frq')
    pitch_contour = pd.read_csv(frq_file_path)['Pitch'].to_numpy()
    
    unit = modifyUnitDurationandPitch(unit, pitch_contour, 
                                      note['Real Duration'], 
                                      note['Real Pitch'])
    
    unit = modifyUnitVolume(unit, note['Volume'])
    
    notes.append(unit)
    
    params = extractAcousticUnitParams(note['Alias'])
    overlaps.append(params[0].values[0])
    
overlaps = np.array(overlaps)

# %% Units Concatenation

CROSS_TOL = 0.99 # last sample of fade-in function
#BASE_OVERLAP = 50

def crossFadeFunction(overlap, tolerance): # fade-in definition
    s_overlap = np.floor((overlap/1000)*SR) # number of overlaping samples
    n = np.arange(s_overlap)
    x0 = s_overlap/2
    alpha = np.log(tolerance/(1 - tolerance))*2/s_overlap
    return(1/(1 + np.exp(alpha*(n - x0))))

# %% Isolated Run-Once Boolean Variable

APPEND_ONCE = False

# %% Preparing overlaps Array for Cross-over Algorithm

overlaps[0] = 0 # there's no unit to overlap before the first

if not APPEND_ONCE:
    overlaps = np.append(overlaps, [0]) # no unit to overlap after the last
    APPEND_ONCE = True
    sample_overlaps = [int(np.floor((o/1000.0)*SR)) for o in overlaps]

# Incializaing Output Array

song_length = 0

for i in range(len(notes)):
    song_length += len(notes[i]) # length of each note
    
song_length -= np.sum(sample_overlaps)# discard used overlaps

song = np.zeros(shape = (int(song_length), ))

# %% Cross-over Algorithm

CROSS_ON = True

c_t = 0 # current time

if CROSS_ON:

    for j in range(len(notes)):
        note_size = len(notes[j])
    
        left_overlap = c_t + sample_overlaps[j] # end of left_overlap
        fade_in = np.flip(crossFadeFunction(overlaps[j], CROSS_TOL))
    
        right_overlap = c_t + note_size - sample_overlaps[j + 1] # beginning of right overlap
        fade_out = crossFadeFunction(overlaps[j + 1], CROSS_TOL)
    
        no_overlap = note_size - sample_overlaps[j] - sample_overlaps[j + 1]
        core = np.ones(shape = (no_overlap,))
        envelope = np.concatenate((fade_in, core, fade_out))
    
        post_note = np.multiply(notes[j], envelope)
    
        left_fill = np.zeros(shape = (c_t,))
        right_fill = np.zeros(shape = (song_length 
                                   - (c_t + note_size),))
        fill_post_note = np.concatenate((left_fill, post_note, right_fill))
    
        plt.subplot(len(notes), 1, j + 1)
        plt.plot(np.concatenate((left_fill, envelope, right_fill)))
    
        song += fill_post_note
    
        c_t += (note_size - sample_overlaps[j + 1])

else:
    first_note = notes[0]
    song = first_note
    #plt.subplot(len(notes), 1, 1)
    #plt.xlim([-5000, 150000])
    #plt.plot(song)
    for j in range(len(notes) - 1):
        song = np.concatenate((song, notes[j + 1]))
        #plt.subplot(len(notes), 1, j + 2)
        #plt.xlim([-5000, 150000])
        #plt.plot(song)
            
#plt.savefig('crossfade.png')
    
# %% Save Song

wv.write(INTER_FILE_NAME[0:-4] + '.wav', SR, np.int16(32767*song))
    
# %% Play Song

sd.play(song, SR)

# %% Tests

TEST_UNIT = extractAcousticUnit('vo')
frq_file_path = VOICEBANK_PATH.joinpath(note['Alias'] + '.frq')
pitch_contour = pd.read_csv(frq_file_path)['Pitch'].to_numpy()

ref_pitch = np.power(2,(np.float32(VB_PITCH_KEY)/12))*PITCH_REF

#plt.subplot(3,2,1)
#plt.xlim([-1000,50000])
#plt.ylim([-0.2,0.2])
#plt.title('Sem modificações')
#plt.plot(TEST_UNIT/50000)

HIGHER_UNIT = modifyUnitDurationandPitch(TEST_UNIT, pitch_contour, 
                                  len(TEST_UNIT)/48, 
                                  1.5*ref_pitch)/1.5

#plt.subplot(3,2,4)
#plt.xlim([5000,7500])
#plt.ylim([-0.2,0.2])
#plt.title('Frequência aumentada')
#plt.plot(HIGHER_UNIT)

LONGER_UNIT = modifyUnitDurationandPitch(TEST_UNIT, pitch_contour, 
                                  1.5*len(TEST_UNIT)/48, 
                                  ref_pitch)/1.5

#plt.subplot(3,2,6)
#plt.xlim([-1000,50000])
#plt.ylim([-0.2,0.2])
#plt.title('Duração extendida')
#plt.plot(LONGER_UNIT)

STRONGER_UNIT = modifyUnitVolume(TEST_UNIT, 1.5)

#plt.subplot(3,2,3)
#plt.xlim([-1000,50000])
#plt.ylim([-0.2,0.2])
#plt.title('Volume aumentado')
#plt.plot(STRONGER_UNIT/50000)

WEAKER_UNIT = modifyUnitVolume(TEST_UNIT, 0.5)

#plt.subplot(3,2,5)
#plt.xlim([-1000,50000])
#plt.ylim([-0.2,0.2])
#plt.title('Volume diminuido')
#plt.plot(WEAKER_UNIT/50000)

HIGHER_LONGER_UNIT = modifyUnitDurationandPitch(TEST_UNIT, pitch_contour, 
                                  1.5*len(TEST_UNIT)/48, 
                                  1.5*ref_pitch)/1.5

#plt.subplot(3,2,2)
#plt.xlim([5000,7500])
#plt.ylim([-0.2,0.2])
#plt.title('Frequência original')
#plt.plot(TEST_UNIT/50000)

#plt.subplots_adjust(left=0.1,
                    #bottom=0.1, 
                    #right=0.9, 
                    #top=0.9, 
                    #wspace=0.4, 
                    #hspace=1)

#plt.savefig('mods.png')

# %% Tests 2

FS, ALL = wv.read( VOICEBANK_PATH.joinpath('v6_ve_vi_vo_vu_v6.wav'))
THIS_UNIT = extractAcousticUnit('vo')
THIS_SIZE = len(ALL)
THIS_INFO = info = df_oto.loc[df_oto['Alias'] == 'vo']
THIS_OFF = int(FS*np.single(info['Offset'].values[0])/1000)
THIS_CUT = int(FS*np.single(info['Cutoff'].values[0])/1000)
THIS_LEFT_FILL = np.zeros(shape = (THIS_OFF,))
THIS_RIGHT_FILL = np.zeros(shape = (THIS_CUT,))
THIS_UNIT_IN_PLACE = np.concatenate((THIS_LEFT_FILL, THIS_UNIT, THIS_RIGHT_FILL))

#plt.subplot(2,1,1)
#plt.ylim([-0.15,0.15])
#plt.title('Arquivo contendo múltiplas unidades acústicas')
#plt.plot(ALL/50000)

#plt.subplot(2,1,2)
#plt.ylim([-0.15,0.15])
#plt.title('Unidade acústica selecionada')
#plt.plot(THIS_UNIT_IN_PLACE/50000)

#plt.subplots_adjust(left=0.1,
                    #bottom=0.1, 
                    #right=0.9, 
                    #top=0.9, 
                    #wspace=0.4, 
                    #hspace=0.5)

#plt.savefig('extract.png')
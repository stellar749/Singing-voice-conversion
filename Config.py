import configparser

#Set the configuration for the vocal separationi
def set_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)

    # Logging
    config_get(config, 'logging', 'logfile', 'log.txt') #logging filename
    config_get(config, 'logging', 'loglevel', 'INFO')  #logging level

    # Dataset
    config_get(config, 'data', 'traindir', "MIR-1K/Wavfile") #Training directory
    config_get(config, 'data', 'testdir', "MIR-1K/UndividedWavfile") #We will get window size / 2 + 1 frequency bins to work with. 1024-1568 seems to be the perfect vales.

    config_get(config, 'data', 'sample_size', "44100") #Sample rate of the audio we will work with.
    config_get(config, 'data', 'window_size', "2048") #Window size / 2 + 1 frequency bins to work obtained
    config_get(config, 'data', 'hop_length', "512") #Size of each bin = hop size / sample size (in ms). The smaller it is, the more bins we get, but we don't need that much resolution.
    config_get(config, 'data', 'sample_length', "20") #Dictates how many frequency bins we give to the neural net for context. Less samples means more guesswork from the network, but also more samples from each song.

    config_get(config, 'model', 'save_history', "true") #Saves  accuracy and loss history per epoch
    config_get(config, 'model', 'history_filename', "history.csv")

    with open(filename, 'w') as configfile: # If the file didn't exist, write default values to it
        config.write(configfile)
    return config

def config_get(config, section, key, default):
    try:
        config.get(section, key)
    except configparser.NoSectionError:
        config.add_section(section)
        config.set(section, key, default)
    except configparser.NoOptionError:
        config.set(section, key, default)
        

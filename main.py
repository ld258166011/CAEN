from imtd import read_data_sets
from configparser import ConfigParser
from tensorflow.python.keras._impl.keras.callbacks import CSVLogger
from models import ConvReconster, Encoder, Reconstructor, Categorizer
from figure import RecDrawer, LatDrawer, LosDrawer
from evaluate import eval_matrics

# Datasets
datasets = read_data_sets('IMTD17', reshape=False)
x_s1train = datasets.s1train.images
x_s2train = datasets.s2train.images
y_s2train = datasets.s2train.labels
x_test = datasets.test.images
y_test = datasets.test.labels

# Configs
config = ConfigParser()
config.read('config')

# Conv Pre-Training
if config.getboolean('CONV', 'load'):
    ConvReconster.load_weights('weights/conv.h5')
else:
    if not config.getint('CONV', 'epochs') == 0:
        ConvReconster.compile(config.get('CONV', 'optimizer'),
                            config.get('CONV', 'loss'))
        ConvReconster.fit(x_s1train, x_s1train, verbose=2,
                       batch_size=config.getint('CONV', 'batch'),
                       epochs=config.getint('CONV', 'epochs'),
                       callbacks=[CSVLogger('losses/conv.csv'),
                                  RecDrawer(x_s2train,
                                            'figures/reconstruct',
                                            suffix='Conv'),
                                  LosDrawer('losses/conv.csv',
                                            'figures/conv_losses')])
        ConvReconster.save_weights('weights/conv.h5')
    
# MLP Pre-Training
if config.getboolean('MLP', 'load'):
    Reconstructor.load_weights('weights/mlp.h5')
else:
    if not config.getint('MLP', 'epochs') == 0:
        for i in range(1, len(ConvReconster.layers)):
            ConvReconster.layers[i].trainable = False
        Reconstructor.compile(config.get('MLP', 'optimizer'),
                              config.get('MLP', 'loss'))
        Reconstructor.fit(x_s1train, x_s1train, verbose=2,
                          batch_size=config.getint('MLP', 'batch'),
                          epochs=config.getint('MLP', 'epochs'),
                          callbacks=[CSVLogger('losses/mlp.csv'),
                                     RecDrawer(x_s2train,
                                               'figures/reconstruct',
                                               suffix='MLP'),
                                     LosDrawer('losses/mlp.csv',
                                               'figures/mlp_losses')])
        for i in range(1, len(ConvReconster.layers)):
            ConvReconster.layers[i].trainable = True
        Reconstructor.save_weights('weights/mlp.h5')

# S1 Training
if config.getboolean('REC', 'load'):
    Reconstructor.load_weights('weights/reconstructor.h5')
else:
    if not config.getint('REC', 'epochs') == 0:
        Reconstructor.compile(config.get('REC', 'optimizer'),
                              config.get('REC', 'loss'))
        Reconstructor.fit(x_s1train, x_s1train, verbose=2,
                          batch_size=config.getint('REC', 'batch'),
                          epochs=config.getint('REC', 'epochs'),
                          callbacks=[CSVLogger('losses/reconstructor.csv'),
                                     RecDrawer(x_s2train,
                                               'figures/reconstruct',
                                               freq=50),
                                     LatDrawer(Encoder, x_test, y_test,
                                               'figures/latent'),
                                     LosDrawer('losses/reconstructor.csv',
                                               'figures/rec_losses')])
        Reconstructor.save_weights('weights/reconstructor.h5')

# S2 Training
if config.getboolean('CAT', 'load'):
    Categorizer.load_weights('weights/categorizer.h5')
else:
    if not config.getint('CAT', 'epochs') == 0:
        Categorizer.compile(config.get('CAT', 'optimizer'),
                           config.get('CAT', 'loss'))
        Categorizer.fit(x_s2train, y_s2train, verbose=2,
                       batch_size=config.getint('CAT', 'batch'),
                       epochs=config.getint('CAT', 'epochs'),
                       callbacks=[CSVLogger('losses/categorizer.csv'),
                                  LosDrawer('losses/categorizer.csv',
                                            'figures/cat_losses',
                                            mode='semilogy')])
        Categorizer.save_weights('weights/categorizer.h5')

# Testing
y_predict = Categorizer.predict(x_test)
eval_matrics(y_predict, y_test,
             'result.txt')

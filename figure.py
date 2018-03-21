from tensorflow.python.keras._impl.keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy
import csv

class RecDrawer(Callback):
    
    def __init__(self, x, filename, suffix=None, freq=10):
        self.x = x
        self.filename = filename
        if not suffix is None:
            self.filename += '_' + suffix
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == (self.freq - 1):
            x_reconstruct = self.model.predict(self.x[:11])
            filename = self.filename + '_' + str(epoch+1)
            print('Drawing', filename)
            chosen_index = (10, 0, 1)
            draw_num = 0
            plt.figure(figsize=(8, 8))
            for i in chosen_index:
                x_ori = self.x[i].reshape(28, 36)
                x_rec = x_reconstruct[i].reshape(28, 36)
                numpy.save("%s_n%d" % (filename, i), x_rec)
                plt.subplot(3, 2, 2 * draw_num + 1)
                plt.imshow(x_ori, cmap="gray")
                plt.xticks([])
                plt.yticks([])
                if draw_num == 0:
                    plt.title("Original")
                plt.subplot(3, 2, 2 * draw_num + 2)
                plt.imshow(x_rec, cmap="gray")
                plt.xticks([])
                plt.yticks([])
                if draw_num == 0:
                    plt.title("Reconstruction")
                draw_num += 1
            plt.tight_layout()
            plt.savefig(filename, dpi=1000)
            plt.close()

class LatDrawer(Callback):
    
    def __init__(self, encoder, x, y, filename, freq=10):
        self.encoder = encoder
        self.x = x
        self.y = y
        self.filename = filename
        self.freq = freq
    
    def on_epoch_end(self, epoch, logs=None):
        layers = self.model.layers
        for i in range(len(layers)):
            if layers[i].name == 'latent':
                if not layers[i].units == 2:
                    return
                break
        if epoch % self.freq == (self.freq - 1):
            encoded = self.encoder.predict(self.x)
            filename = self.filename + '_' + str(epoch+1)
            numpy.save(filename, encoded)
            print('Drawing', filename)
            class_marker = ['^', 'v', '<', '>', 's', 'D', 'p', 'h', '8', 'P', 'X', '*', 'o']
            color = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple',
                     'olive', 'brown', 'lime', 'fuchsia', 'mediumspringgreen', 'whitesmoke']
            plt.figure(figsize=(4, 3.2))
            bg = self.y.sum(1) == 0
            count = len(bg[bg])
            size = numpy.ones(count) * 15
            marker = class_marker[12]
            plt.scatter(encoded[bg, 0], encoded[bg, 1],
                        s=size, c=color[12], alpha=0.3, marker=marker,
                        lw=0.5, edgecolors='black')
            dots = list()
            labels = list()
            for i in range(12):
                app = self.y[:, i] == 1
                count = len(app[app])
                size = numpy.ones(count) * 60
                marker = class_marker[i]
                plt.scatter(encoded[app, 0], encoded[app, 1],
                            s=size, c=color[i], alpha=0.5, marker=marker,
                            lw=0.5, edgecolors='black')
                dot, = plt.plot(100, 100, lw=0, marker=class_marker[i],
                                markerfacecolor=color[i], markeredgecolor='black',
                                markeredgewidth=0.5, markersize=7.5)
                dots.append(dot)
                labels.append(str(i))
            dot, = plt.plot(100, 100, lw=0, marker=class_marker[12],
                            markerfacecolor=color[12], markeredgecolor='black',
                            markeredgewidth=0.5, markersize=4)
            dots.append(dot)
            labels.append(str(12))
            plt.xlim([-40, 40])
            plt.ylim([-40, 40])
            plt.xlabel("z (axis 0)")
            plt.ylabel("z (axis 1)")
            plt.grid(linestyle=":", linewidth=0.5)       
            plt.subplots_adjust(left=0.15, right=0.8, bottom=0.15, top=0.95, wspace=0, hspace=0)
            plt.legend(dots, labels, title='Labels', markerfirst=False,
                       loc=1, bbox_to_anchor=(1.26, 1),
                       labelspacing=0.31, borderaxespad=0., handletextpad=0.1)
            plt.savefig(filename, dpi=500)
            plt.close()

class LosDrawer(Callback):
    
    def __init__(self, scvname, pngname, mode='plot'):
        self.scvname = scvname
        self.pngname = pngname
        self.mode = mode
    
    def on_train_end(self, logs=None):
        csvFile = open(self.scvname)
        csvDict = csv.DictReader(csvFile)
        print('Drawing', self.pngname)
        epochs = list()
        losses = list()
        for item in csvDict:
            epochs.append(int(item['epoch']))
            losses.append(float(item['loss']))
        plt.figure(figsize=(4, 3))
        if self.mode == 'plot':
            plt.plot(epochs, losses, linewidth=1)
        elif self.mode == 'semilogy':
            plt.semilogy(epochs, losses, basey = 2, linewidth=1)
        plt.xlabel('Training Epoch')
        plt.ylabel('Model Loss')
        plt.grid(axis="y", linestyle=":", linewidth=1)
        plt.tight_layout()
        plt.savefig(self.pngname, dpi=500)
        plt.close()        

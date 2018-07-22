import sys
import glob
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == "__main__":

    target_dir = 'result'
    
    fig = plt.figure()

    ims = []

    flist = ['{}/{}.png'.format(target_dir, i) for i in range(0, 200)]
    for f in flist:
        image = misc.imread(f)
        if image is None: continue
        ims.append((plt.imshow(image),))

    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=100, blit=True)

    ani.save('foo.gif', writer='imagemagick')

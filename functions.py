
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio as imageio

#for pulling data from the Boss
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import ChannelResource
import matplotlib.pyplot as plt
import numpy as np

#pulls data from Boss
boss = BossRemote({
    "protocol": "https",
    "host": "api.theboss.io",
    "token": "ada424f4d529a6c352f56d6064a317fa8bda7821"
})

#Here you will specify form where the data is coming from, the resolution, and the size of your image.
volume = boss.get_cutout(
    boss.get_channel("em","kasthuri2015", "ac4"), 0,
    [0, 200], [0, 200], [0, 20],)

print(volume)
plt.imshow(volume[1,:,:], cmap= "gray")
plt.show()

#saves and loads data
np.save('z_coord_' + str(20) + '.npy', volume[0,:,:])
data20 = 'z_coord_20.npy'
data20 = np.load(data20)

#plot function # TODO: call in @app.route
def plot(im1, im2=None, cmap1='gray', cmap2='jet', slice=0,
         alpha=1, show_plot=True, save_plot=False):
    """
    Convenience function to handle plotting of neurodata arrays.
    Mostly tested with 8-bit image and 32-bit annos, but lots of
    things should work.  Mimics (and uses) matplotlib, but transparently
    deals with RAMON objects, transposes, and overlays.  Slices 3D arrays
    and allows for different blending when using overlays.  We require
    (but verify) that dimensions are the same when overlaying.
    Arguments:
        im1 (array): RAMONObject or numpy array
        im2 (array) [None]:  RAMONObject or numpy array
        cmap1 (string ['gray']): Colormap for base image
        cmap2 (string ['jet']): Colormap for overlay image
        slice (int) [0]: Used to choose slice from 3D array
        alpha (float) [1]: Used to set blending option between 0-1
    Returns:
        None.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # get im1_proc as 2D array
    fig = plt.figure()
    # fig.set_size_inches(2, 2)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    base_image = False
    im1_proc = None

    if hasattr(im1, 'cutout') and im1.cutout is not None:
        im1_proc = im1.cutout
    elif im1 is not None:
        im1_proc = im1

    if im1_proc is not None and len(np.shape(im1_proc)) == 3:
        im1_proc = im1_proc[:, :, slice]

    if im1_proc is not None:
        base_image = True

    # get im2_proc as 2D array if exists
    overlay_image = False
    im2_proc = None

    if im2 is not None:

        if hasattr(im2, 'cutout') and im2.cutout is not None:
            im2_proc = im2.cutout
        elif im2 is not None:
            im2_proc = im2

        if im2_proc is not None and len(np.shape(im2_proc)) == 3:
            im2_proc = im2_proc[:, :, slice]

    if im2_proc is not None and np.shape(im1_proc) == np.shape(im2_proc):
        overlay_image = True

    if base_image:

        plt.imshow(im1_proc.T, cmap=cmap1, interpolation='bilinear')

    if base_image and overlay_image and alpha == 1:
        # This option is often recommended but seems less good in general.
        # Produces nice solid overlays for things like ground truth
        im2_proc = np.ma.masked_where(im2_proc == 0, im2_proc)
        plt.imshow(im2_proc.T, cmap=cmap2, interpolation='nearest')

    elif base_image and overlay_image and alpha < 1:

        plt.hold(True)
        im2_proc = np.asarray(im2_proc, dtype='float')  # TODO better way
        im2_proc[im2_proc == 0] = np.nan  # zero out bg
        plt.imshow(im2_proc.T, cmap=cmap2,
                   alpha=alpha, interpolation='nearest')

    if save_plot is not False:
        # TODO: White-space
        plt.savefig(save_plot, dpi=300, pad_inches=0)

    if show_plot is True:
        plt.show()

    pass

#save function # TODO: call in @app.route
def save_movie(im1, im2=None, cmap1='gray', cmap2='jet', alpha=1,
               fps=1, outFile='test.mp4'):

    # TODO properly nest plot function

    import moviepy.editor as mpy
    from moviepy.video.io.bindings import mplfig_to_npimage
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')

    #changed from time = list(range(0, int(np.shape(im1)[2])))
    import matplotlib.pyplot as plt
    time = list(range(0, int(np.shape(im1)[1])))

    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    def animate(slice):

        import numpy as np
        import matplotlib.pyplot as plt
        # get im1_proc as 2D array

        base_image = False
        im1_proc = None

        if hasattr(im1, 'cutout') and im1.cutout is not None:
            im1_proc = im1.cutout
        elif im1 is not None:
            im1_proc = im1

        if im1_proc is not None and len(np.shape(im1_proc)) == 3:
            im1_proc = im1_proc[:, :, slice]

        if im1_proc is not None:
            base_image = True

        # get im2_proc as 2D array if exists
        overlay_image = False
        im2_proc = None

        if im2 is not None:

            if hasattr(im2, 'cutout') and im2.cutout is not None:
                im2_proc = im2.cutout
            elif im2 is not None:
                im2_proc = im2

            if im2_proc is not None and len(np.shape(im2_proc)) == 3:
                im2_proc = im2_proc[:, :, slice]

        if im2_proc is not None and np.shape(im1_proc) == np.shape(im2_proc):
            overlay_image = True

        if base_image:

            plt.imshow(im1_proc.T, cmap=cmap1, interpolation='bilinear')

        if base_image and overlay_image and alpha == 1:
            # This option is often recommended but seems less good in general.
            # Produces nice solid overlays for things like ground truth
            im2_proc = np.ma.masked_where(im2_proc == 0, im2_proc)
            plt.imshow(im2_proc.T, cmap=cmap2, interpolation='nearest')

        elif base_image and overlay_image and alpha < 1:

            plt.hold(True)
            im2_proc = np.asarray(im2_proc, dtype='float')  # TODO better way
            im2_proc[im2_proc == 0] = np.nan  # zero out bg
            plt.imshow(im2_proc.T, cmap=cmap2,
                       alpha=alpha, interpolation='nearest')

        return mplfig_to_npimage(fig)

    animation = mpy.VideoClip(animate, duration=len(time))

    import os.path
    extension = os.path.splitext(outFile)[1]

    if extension is 'gif':
        animation.write_gif(outFile, fps=fps, fuzz=0)

    else:  # 'mp4'
        animation.write_videofile(outFile, fps=fps, bitrate='5000k',
                                  codec='libx264')

#save created movie
imageio.plugins.ffmpeg.download()
save_movie(data20)

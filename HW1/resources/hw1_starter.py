# This starter code requires functions in the Dolly Zoom Notebook to work
import numpy as np

from dolly_zoom import *

import os
import imageio




# Call this function to generate gif. make sure you have rotY() implemented.
def generate_gif():
    n_frames = 30
    if not os.path.isdir("frames"):
        os.mkdir("frames")
    fstr = "frames/%d.png"
    for i, theta in enumerate(np.arange(0, 2 * np.pi, 2 * np.pi / n_frames)):
        fname = fstr % i
        renderCube(f=15, t=(0, 0, 3), R=rotY(theta))
        plt.savefig(fname)
        plt.close()

    with imageio.get_writer("cube.gif", mode='I') as writer:
        for i in range(n_frames):
            frame = plt.imread(fstr % i)
            # Fixme:
            frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)
            os.remove(fstr % i)

    os.rmdir("frames")


# Problem 1.(a)
def rotY(theta_radian):
    # takes an angle theta (in radian) and outputs the 3D rotation matrix of
    # rotating by theta about the y-axis (right-hand rule).
    # theta_radian: R=np.eye(3)
    return np.array([
        [ np.cos(theta_radian), 0, np.sin(theta_radian)],
        [ 0,                    1, 0                   ],
        [-np.sin(theta_radian), 0, np.cos(theta_radian)]
    ])


# Problem 1.(b)
from tqdm import tqdm


def rotX(theta_radian):
    return np.array([
        [1, 0,                     0],
        [0, np.cos(theta_radian), -np.sin(theta_radian)],
        [0, np.sin(theta_radian),  np.cos(theta_radian)]
    ])


def rot_X_then_Y():
    n_frames = 30
    if not os.path.isdir("rot_X_then_Y"):
        os.mkdir("rot_X_then_Y")
    fstr_x = "rot_X_then_Y/x_%d.png"
    fstr_y = "rot_X_then_Y/y_%d.png"

    # rotating theta/4 about X-axis
    for i, theta in enumerate(np.arange(0, np.pi / 4, (np.pi / 4) / n_frames)):
        fname = fstr_x % i
        renderCube(f=15, t=(0, 0, 3), R=rotX(theta))  # only rotate about X-axis fistly.
        plt.savefig(fname)
        plt.close()

    # then, rotating theta/4 about Y-axis
    for i, theta in enumerate(np.arange(0, np.pi / 4, (np.pi / 4) / n_frames)):
        fname = fstr_y % i
        R = rotX(np.pi / 4) @ rotY(theta)  # the cube has already rotated pi/4 about X-axis
        renderCube(f=15, t=(0, 0, 3), R=R)
        plt.savefig(fname)
        plt.close()

    # combine every frames into one gif
    with imageio.get_writer("rot_X_then_Y.gif", mode='I') as writer:
        for i in tqdm(range(n_frames), desc="Generating rot_X frames"):
            frame = plt.imread(fstr_x % i)
            frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)

        for i in tqdm(range(n_frames), desc="Generating rot_Y frames"):
            frame = plt.imread(fstr_y % i)
            frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)

        # delete temp files and dir.
        for i in range(n_frames):
            os.remove(fstr_x % i)
            os.remove(fstr_y % i)

    os.rmdir("rot_X_then_Y")

def rot_Y_then_X():
    n_frames = 30

    if not os.path.isdir("rot_Y_then_X"):
        os.mkdir("rot_Y_then_X")

    fstr_y = "rot_Y_then_X/y_%d.png"
    fstr_x = "rot_Y_then_X/x_%d.png"  # 修正文件名

    for i, theta in enumerate(np.arange(0, np.pi / 4, (np.pi / 4) / n_frames)):
        fname = fstr_y % i
        renderCube(f=15, t=(0, 0, 3), R=rotY(theta))    # only rotate about Y-axis fistly.
        plt.savefig(fname)
        plt.close()

    # then, rotating theta/4 about X-axis
    for i, theta in enumerate(np.arange(0, np.pi / 4, (np.pi / 4) / n_frames)):
        fname = fstr_x % i
        R = rotY(np.pi / 4) @ rotX(theta)  # the cube has already rotated pi/4 about X-axis
        renderCube(f=15, t=(0, 0, 3), R=R)
        plt.savefig(fname)
        plt.close()

    with imageio.get_writer("rot_Y_then_X.gif", mode='I') as writer:
        for i in tqdm(range(n_frames), desc="Generating rot_Y frames"):
            frame = plt.imread(fstr_y % i)
            frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)

        for i in tqdm(range(n_frames), desc="Generating rot_X frames"):
            frame = plt.imread(fstr_x % i)
            frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)
        # delete temp files and dir.
        for i in range(n_frames):
            os.remove(fstr_x % i)
            os.remove(fstr_y % i)

    os.rmdir("rot_Y_then_X")


# Problem 1.(c)
def rot_together():
    fname = 'temp_fig.png'

    theta_x = np.pi / 5
    theta_y = np.pi / 4

    R = rotX(theta_x) @ rotY(theta_y)  # the cube has already rotated pi/4 about X-axis
    renderCube(f=15, t=(0, 0, 3), R=R)

    plt.savefig(fname)
    plt.close()


if __name__ == '__main__':
    # generate_gif()
    # rot_X_then_Y()
    # rot_Y_then_X()
    rot_together()

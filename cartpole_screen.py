import torch
import torchvision.transforms as T
import numpy as np

from PIL import Image


resize = T.Compose([T.ToPILImage(), T.Resize(84, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])


def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def capture_frames(env, device, n_frames, action=0):
    screen_buffer = []
    for i in range(n_frames):
        screen = get_screen(env, device, monochrome=True)
        screen_buffer.append(screen)
    return torch.sum(torch.stack(screen_buffer), 0).to(device)


def get_screen(env, device, monochrome=False):
    # Transpose screen to (channels, height, width) format used by torch
    screen = env.render()
    if monochrome:
        screen = np.dot(screen[..., :3], [0.299, 0.587, 0.114])
        screen_height, screen_width = screen.shape
    else:
        screen = screen.transpose((2, 0, 1))
        _, screen_height, screen_width = screen.shape
    
    # Strip out the useless top and bottom parts of the screen
    screen = screen[:, int(screen_height * 0.4): int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)

    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    # Strip off the edges so we have a square image centered on the cart

    if monochrome:
        screen = screen[:, slice_range]
        height, width = screen.shape
    else:
        screen = screen[:, :, slice_range]
        _, height, width = screen.shape

    # Now slice the screen into a square shape to fit our convolutions
    box_slice = slice(width // 2 - height // 2,
                      width // 2 + height // 2)
    if monochrome:
        screen = screen[:, box_slice]
    else:
        screen = screen[:, :, box_slice]

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    # return resize(screen).unsqueeze(0).to(device)
    return resize(screen).to(device)

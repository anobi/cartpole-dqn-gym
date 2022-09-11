import torch
import numpy as np

from PIL import Image, ImageOps

def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def capture_frames(env, device, n_frames, action=0):
    screen_buffer = []
    for i in range(n_frames):
        screen = get_torch_screen(env, device, monochrome=True)
        screen_buffer.append(screen)
    return torch.sum(torch.stack(screen_buffer), 0).to(device)


def get_human_screen(render_frame):
    return Image.fromarray(render_frame)


def get_torch_screen(render_frame, device, image_size, resize_func):
    # Grayscale and resize the screen to 84x84
    p_img = Image.fromarray(render_frame)
    p_img = ImageOps.grayscale(p_img)
    p_img = ImageOps.fit(p_img, [image_size, image_size])
    
    img_array = np.ascontiguousarray(p_img, dtype=np.float32) / 255
    torch_screen = torch.from_numpy(img_array)
    return resize_func(torch_screen).unsqueeze(0).to(device)
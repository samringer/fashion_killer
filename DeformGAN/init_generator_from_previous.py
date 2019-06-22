import torch

from DeformGAN.model.generator import Generator


def init_generator_from_previous(init_path):
    """
    Allow initialisation of generator from a previous model,
    even if that model was trained on smaller imgs.
    """
    new_generator = Generator()

    init_state_dict = torch.load(init_path)
    new_state_dict = new_generator.state_dict()

    for k, v in init_state_dict.items():
        if k in new_state_dict.keys():
            if v.shape == new_state_dict[k].shape:
                new_state_dict.update({k: v})
    new_generator.load_state_dict(new_state_dict)
    return new_generator

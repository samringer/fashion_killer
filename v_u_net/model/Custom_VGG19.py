from torch.utils import model_zoo
from torchvision.models.vgg import VGG, make_layers, cfg, model_urls

class Custom_VGG19(VGG):
    """
    A VGG model that has been altered to
    allow for returning actiations in intermediate
    layers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_layers = [2, 7, 12, 17, 23, 34]

    def forward(self, x):
        loss_layers = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.loss_layers:
                loss_layers.append(x)
        return loss_layers

def get_custom_VGG19(pretrained=True, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = Custom_VGG19(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

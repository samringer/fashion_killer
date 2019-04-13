import torch
from torchvision import transforms

from v_u_net.model.Model import Model
from Fashion_Killer.Appearance_Modules.Localise_Joint_Appearances import get_localised_joints
from Fashion_Killer.Pose_Modules.Drawers import Pose_Drawer
import Fashion_Killer.hyperparams as hp


class Image_Generator():
    """
    Creates new images that combines the appearance of one image
    and the pose of another.
    """
    def __init__(self, model_path, joints_to_localise):
        """
        Args:
            model_path (str): Path to model to use for inference.
            joints_to_localise (l o ints): Integer indicies of the desired joints to localise.
        """
        self.model = self._get_model(model_path)
        self.data_preparer = Data_Preparer()
        
    def _get_model(self, model_path):
        """
        Initialises a model with pretrained weights and puts model
        in eval mode.
        Args:
            model_path (str): Path to model to use for inference.
        Returns:
            model (PyTorch model): The initialised model to use for inference
        """
        model = Model()
        model.load_state_dict(torch.load(model_path))
        model = model.eval()
        if hp.use_cuda_inference:
            model = model.cuda()
        return model
    
    def generate_image(self, app_img, app_img_joint_pos, pose_joint_pos):
        """
        Args:
            app_img (np array): np array of image with desired appearance.
            app_img_joint_pos (l o floats ): List of the coordinates of the joints found by OpenPose from desired appearance image.
            pose_joint_pos (l o floats): List of the coordinates of the joints found by OpenPose for desired pose.
        Returns:
            generated_img (np array): np array of generated image
        """
        prepared_input_data = self.data_preparer.prepare_input_data(app_img, app_img_joint_pos, pose_joint_pos)
        generated_img, _, _, _, _ = self.model(*prepared_input_data)
        generated_img = (generated_img*0.5) + 0.5  # Denormalise
        generated_img = generated_img.squeeze(0).permute(1, 2, 0)
        generated_img = generated_img.detach().cpu().numpy()
        return generated_img


class Data_Preparer:
    """
    Takes two input images and prepares them both for inference.
    """
    def __init__(self):
        self.pose_drawer = Pose_Drawer()
        self.joints_to_localise = joints_to_localise

    def prepare_input_data(self, app_img, app_img_joint_pos, pose_joint_pos):
        """
        Prepares the data for input into the model.
        Returns a tuple of:
            app_img (PyTorch tensor): Tensor of the desired appearance img.
            app_img_pose (PyTorch tensor): Tensor of the pose img of the desired appearance img.
            pose_img (PyTorch tensor): Tensor of the desired pose img.
            localised_joints (PyTorch tensor): Tensor of zoomed images of appearance img joints.
        """
        app_joint_pixels = (app_img_joint_pos*hp.image_edge_size).astype('int')
        localised_joints_list = get_localised_joints(app_img, 
                                                     self.joints_to_localise,
                                                     app_joint_pixels)

        app_img_pose = self.pose_drawer.draw_pose_img(app_img_joint_pos)
        pose_img = self.pose_drawer.draw_pose_img(pose_joint_pos)

        app_img = self._transform_img(app_img)
        app_img_pose = self._transform_img(app_img_pose).float()
        pose_img = self._transform_img(pose_img).float()

        # Need to normalise one by one as lots of the images are black
        localised_joint_list = [self._transform_img(joint_img) for joint_img in localised_joints_list]
        localised_joints = torch.cat(localised_joint_list, dim=0).float()

        # Mock having batch size one to make dimensions work in model.
        app_img = app_img.unsqueeze(0)
        app_img_pose = app_img_pose.unsqueeze(0)
        pose_img = pose_img.unsqueeze(0)
        localised_joints = localised_joints.unsqueeze(0)

        if hp.use_cuda_inference:
            app_img = app_img.cuda()
            app_img_pose = app_img_pose.cuda()
            pose_img = pose_img.cuda()
            localised_joints = localised_joints.cuda()

        return (app_img, app_img_pose, pose_img, localised_joints)


def _transform_img(img):
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    return transform(img)

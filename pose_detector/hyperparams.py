use_cuda = True
use_fp16 = True
overtrain = False
leakiness = 0.2
batch_size = 4
num_joints = 18
num_limbs = 17
min_joints_to_train_on = 10

num_epochs = 10
learning_rate = 1e-4

keypoint_from_heatmap_threshold = 0.8

ts_log_interval = 30
checkpoint_interval = 5000
checkpoint_load_path = None


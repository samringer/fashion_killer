import unittest
import shutil
from tempfile import mkdtemp
from os.path import join, dirname, realpath

from absl import flags, app

from pose_detector.train import train

FUNCTESTS_DIR = dirname(realpath(__file__))


class TestPoseDetectorTrain(unittest.TestCase):
    """
    Test the ability of the Pose Detector train code to run a complete
    forward and backward pass.
    """
    def setUp(self):
        self.tmp_dir = mkdtemp()
        self.args = [
            'prog',
            '--data_dir', join(FUNCTESTS_DIR, 'data/pose_detector'),
            '--num_epochs', '1',
            '--batch_size', '1',
            '--task_path', self.tmp_dir,
            '--exp_name', 'test_pose_detector',
            '--use_cuda', 'False',
            '--use_fp16', 'False',
        ]

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        flags.FLAGS.unparse_flags()

    def test_pose_detector_train(self):
        """
        Test a full training loop.
        """
        with self.assertRaises(SystemExit) as seo:
            app.run(main=train, argv=self.args)
        self.assertEqual(seo.exception.code, None)


if __name__ == "__main__":
    unittest.main()

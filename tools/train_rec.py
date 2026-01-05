import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--eval',
        action='store_true',
        default=True,
        help='Whether to perform evaluation in train',
    )
    parser.add_argument(
        '--task',
        type=str,
        default='rec',
        choices=['rec', 'formula_rec'],
        help='Task type: rec or formula_rec',
    )
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    task_type = FLAGS.get('task', 'rec')
    trainer = Trainer(cfg,
                      mode='train_eval' if FLAGS['eval'] else 'train',
                      task=task_type)
    trainer.train()



if __name__ == '__main__':
    main()

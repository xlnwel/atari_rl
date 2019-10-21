import os, sys
import argparse
import logging
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run.grid_search import GridSearch
from utility.yaml_op import load_args
from utility.debug_tools import assert_colorize
from utility.utils import str2bool


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', '-a',
                        type=str,
                        nargs='*',
                        choices=['duel', 'iqn', 'c51', 'rainbow', 'rainbow-iqn', 
                                'apex-double', 'apex-duel', 'apex-iqn', 'apex-c51', 'apex-rainbow', 'apex-rainbow-iqn'],
                        default='rainbow-iqn')
    parser.add_argument('--environment', '-e',
                        type=str,
                        default=None)
    parser.add_argument('--render', '-r',
                        action='store_true')
    parser.add_argument('--trials', '-t',
                        type=int,
                        default=1,
                        help='number of trials')
    parser.add_argument('--prefix', '-p',
                        default='',
                        help='prefix for model dir')
    parser.add_argument('--checkpoint', '-c',
                        type=str,
                        default='',
                        help='checkpoint path to restore, of form "model_root_dir/model_name"')
    args = parser.parse_args()

    return args

def import_main(algorithm):
    if 'apex' in algorithm:
        from algo.distributed_train import main
    else:
        from algo.single_train import main

    return main
    
def get_arg_file(algorithm):
    if algorithm == 'duel':
        arg_file = 'algo/ddqn/args.yaml'
    elif algorithm == 'rainbow':
        arg_file = 'algo/rainbow/args.yaml'
    elif algorithm == 'rainbow-iqn':
        arg_file = 'algo/rainbow_iqn/args.yaml'
    else:
        raise NotImplementedError(f'In valid name {algorithm}')

    return arg_file

if __name__ == '__main__':
    cmd_args = parse_cmd_args()
    algorithm = cmd_args.algorithm

    processes = []
    for algo in algorithm:
        main = import_main(algo)
        arg_file = get_arg_file(algo)

        if cmd_args.checkpoint != '':
            args = load_args(arg_file)
            env_args = args['env']
            agent_args = args['agent']
            buffer_args = args['buffer'] if 'buffer' in args else {}
            checkpoint = cmd_args.checkpoint
            assert_colorize(os.path.exists(checkpoint), 'Checkpoint does not exists')
            agent_args['model_root_dir'], agent_args['model_name'] = os.path.split(checkpoint)
            predir, _ = os.path.split(agent_args['model_root_dir'])
            agent_args['log_root_dir'] = predir + '/logs'
            env_args['video_path'] = predir + '/video'
            print('model_root_dir', agent_args['model_root_dir'])
            print('model_name', agent_args['model_name'])
            print('root_log_dir', agent_args['log_root_dir'])
            print('video_path', env_args['video_path'])

            main(env_args, agent_args, buffer_args, render=cmd_args.render, restore=True)
        else:
            prefix = cmd_args.prefix
            # Although random parameter search is in general better than grid search, 
            # we here continue to go with grid search since it is easier to deal with architecture search
            gs = GridSearch(arg_file, main, render=cmd_args.render, n_trials=cmd_args.trials, dir_prefix=prefix)

            if algo:
                gs.agent_args['algo'] = algo
            if cmd_args.environment:
                gs.env_args['name'] = cmd_args.environment
            
            # Grid search happens here
            if algo == 'duel':
                processes += gs()
            elif algo == 'rainbow':
                processes += gs(Qnets=dict(fixup=[True, False]))
            elif algo == 'rainbow-iqn':
                processes += gs()

    [p.join() for p in processes]
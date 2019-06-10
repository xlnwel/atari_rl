import os, sys
import argparse
import logging
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run.grid_search import GridSearch
from utility.yaml_op import load_args
from utility.utils import assert_colorize


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', '-a',
                        type=str,
                        choices=['rainbow-iqn'])
    parser.add_argument('--render', '-r',
                        type=str,
                        choices=['true', 'false'],
                        default='false')
    parser.add_argument('--trials', '-t',
                        type=int,
                        default=1,
                        help='number of trials')
    parser.add_argument('--prefix', '-p',
                        default='',
                        help='prefix for model dir')
    parser.add_argument('--file', '-f',
                        type=str,
                        default='',
                        help='filepath to restore')
    args = parser.parse_args()

    return args

def import_main(algorithm):
    if algorithm == 'rainbow-iqn':
        from run.single_train import main
    else:
        raise NotImplementedError

    return main
    
def get_arg_file(algorithm):
    if algorithm == 'rainbow-iqn':
        arg_file = 'algo/rainbow_iqn/args.yaml'
    else:
        raise NotImplementedError

    return arg_file

if __name__ == '__main__':
    cmd_args = parse_cmd_args()
    algorithm = cmd_args.algorithm
    
    arg_file = get_arg_file(algorithm)
    main = import_main(algorithm)
    
    render = True if cmd_args.render == 'true' else False

    if cmd_args.file != '':
        args = load_args(arg_file)
        env_args = args['env']
        agent_args = args['agent']
        buffer_args = args['buffer'] if 'buffer' in args else {}
        model_file = cmd_args.file
        assert_colorize(os.path.exists(model_file), 'Model file does not exists')
        agent_args['model_root_dir'], agent_args['model_name'] = os.path.split(model_file)
        agent_args['log_root_dir'], _ = os.path.split(agent_args['model_root_dir'])
        agent_args['log_root_dir'] += '/logs'

        main(env_args, agent_args, buffer_args, render=render)
    else:
        prefix = cmd_args.prefix
        if algorithm.startswith('apex'):
            prefix = f'{prefix}-apex' if prefix else 'apex'
        # Although random parameter search is in general better than grid search, 
        # we here continue to go with grid search since it is easier to deal with architecture search
        gs = GridSearch(arg_file, main, render=render, n_trials=cmd_args.trials, dir_prefix=prefix)

        # Grid search happens here
        if algorithm == 'rainbow-iqn':
            gs()
        else:
            raise NotImplementedError

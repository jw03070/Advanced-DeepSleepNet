
import time
import logging
logging.basicConfig(level=logging.WARNING)

import argparse
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB # BOHB 라이브러리

import numpy as np
import random
import warnings
import json

start = time.time()
np.random.seed(7)
random.seed(7)
host = '127.0.0.1'

parser = argparse.ArgumentParser(description='BOHB - CNN')
parser.add_argument('--min_budget', type=float, help='Minimum number of epochs for training.', default=1)
parser.add_argument('--max_budget', type=float, help='Maximum number of epochs for training.', default=9)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=16)
parser.add_argument('--shared_directory', type=str,
                    help='A directory that is accessible for all processes, e.g. a NFS share.',
                    default='./logs/BOHB')
parser.add_argument('--backend',
                    help='Toggles which worker is used. Choose between a pytorch and a keras implementation.',
                    choices=['pytorch', 'keras'], default='keras')
parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=1)
parser.add_argument('--previous_run_dir',type=str, help='A directory that contains a config.json and results.json for the same configuration space.', default='./logs/BOHB')
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')


args = parser.parse_args()

if args.backend == 'pytorch':
    from pytorch_worker import PyTorchWorker as worker
else:
    from cnn_bohb import KerasWorker as worker

result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=True) # True or False

# Start a nameserver:
NS = hpns.NameServer(run_id='Sleepstage', host=host, port=None, working_directory=args.shared_directory)
NS.start()



workers=[] # 멀티 쓰레드 설정가능
for i in range(args.n_workers):
    w = worker(epoch=10, nameserver=host, run_id='Sleepstage', id=i)
    w.load_nameserver_credentials(working_directory=args.shared_directory)
    w.run(background=True)
    workers.append(w)

print("CPU thread is {0}".format(len(workers)))

# # #Run an optimizer
bohb = BOHB(configspace=w.get_configspace(),
            run_id='Sleepstage',
            result_logger=result_logger,
            min_budget=args.min_budget, max_budget=args.max_budget,
            )
res = bohb.run(n_iterations=args.n_iterations,min_n_workers=args.n_workers)

id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()
all_runs = res.get_all_runs()

bohb.shutdown(shutdown_workers=True)
NS.shutdown()

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

print("실행 시간:", time.time() - start)
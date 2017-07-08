import os, sys, datetime, argparse, yaml
import contextlib
from collections import OrderedDict as ordict
import itertools as it
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pt, models, util
from holster import H
import data

parser = argparse.ArgumentParser()
parser.add_argument("--hp")
parser.add_argument("--hpfile")
parser.add_argument("--basename")
parser.add_argument("--num_epochs", type=int)
parser.add_argument("--resume", type=util.bouillon)
parser.add_argument("--runs-dir", default=os.getcwd())

def main():
  args = parser.parse_args()

  config = H()
  config.basename = args.basename
  config.resume = args.resume
  config.num_epochs = args.num_epochs
  config.runs_dir = args.runs_dir

  assert args.hpfile
  with open(args.hpfile) as hpfile:
    config.hp = yaml.load(hpfile)

  print(config.hp)
  config.label = util.make_label(config)
  run_dir = "%s_%s" % (datetime.datetime.now().isoformat(), config.label)
  run_dir = run_dir[:255] # >:-(((((((((((((((((((((((((((((((((((((((((
  config.run_dir = os.path.join(config.runs_dir, run_dir)

  config.save_path = os.path.join(config.run_dir, "parameters.npz")

  config.data = data.make(config.hp.data.kind, config.hp.data)
  config.hp.data_dim = config.data.data_dim
  config.hp.model.data_dim = config.data.data_dim

  # NOTE: all hyperparameters must be set at this point
  prepare_run_directory(config)

  train(config)


def train(config):
  with pt.parameters() as pp:
    if os.path.exists(config.save_path):
      print("resuming from", config.save_path)
      pp.load_from(config.save_path)

    # same architecture and parameters but different state
    train_model = models.make(config.hp.model.kind, config.hp.model)
    sample_model = models.make(config.hp.model.kind, config.hp.model)
  
    # TODO probably just use a separate thread with timer to do this at controllable intervals
    def after_step_hook():
      pp.save_to(config.save_path)
      if np.random.random() < 1:
        with util.timing("sample"):
          sample()

    def sample():
      print("sampling")
      length = 50
      cond = next(util.batches(config.data.examples.valid, batch_size=10))
      cond = next(util.segments(cond, length=sample_model.optimal_cutoff - length))
      pred = sample_model.sample(cond, length=length)
      for condex, predex in zip(cond, pred):
        condex = condex[-length:]
        print(repr(config.data.decode(condex)), "->", repr(config.data.decode(predex)))

    # evaluate model once to ensure parameters are known
    sample_model.sample(next(util.segments(config.data.examples.train[:10],
                                           length=sample_model.optimal_cutoff)),
                        length=sample_model.optimal_cutoff)
    with pp.frozen():
      try:
        optimizer = torch.optim.Adam(list(pp), lr=1e-3)
        for epoch in it.count(0):
          if config.num_epochs is not None and not (epoch < config.num_epochs):
            break
          for batch in util.batches(config.data.examples.train, config.hp.batch_size):
            train_model.train(batch, parameters=pp, optimizer=optimizer,
                              after_step_hook=after_step_hook)
      except KeyboardInterrupt:
        after_step_hook()
        raise
  
  import pdb; pdb.set_trace()

def prepare_run_directory(config):
  # FIXME instead make a flag resume_from, load hyperparameters from there
  if not config.resume:
    if os.path.exists(config.run_dir):
      shutil.rmtree(config.run_dir)
  if not os.path.exists(config.run_dir):
    os.makedirs(config.run_dir)
  if not config.resume:
    with open(os.path.join(config.run_dir, "hp.conf"), "w") as f:
      f.write(util.serialize_hp(config.hp, outer_separator="\n"))

if __name__ == "__main__":
  main()

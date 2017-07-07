import tempfile, yaml
import subprocess as sp
from holster import H
import util

hp = H()

hp["model.kind"] = "wayback"
hp.model.strides = [1, 8, 64]
hp.model.cutoffs = [64] * len(hp.model.strides)
hp.model.layers = [H(kind="lstm", size=100, activation="logselu", affinity="weightnorm")
                   for _ in hp.model.strides]
hp.model.vskip = 1

hp.batch_size = 10

hp["data.kind"] = "enwik8"
hp.data.directory = "/data/lisatmp4/cooijmat/datasets/enwik8"

with tempfile.NamedTemporaryFile(mode="w") as hpfile:
  yaml.dump(hp, hpfile)
  sp.check_call("ipython3 --pdb -- /u/cooijmat/dev/wayback/pt/train.py".split()
                + ["--hpfile", hpfile.name])

#source activate pytorch
#ipython3 --pdb $HOME/dev/wayback/pt/train.py --hpfile hppath

# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# ruff: noqa
# pylint: disable=all
# mypy: ignore-errors
# type: ignore

__version__ = "0.2.2"

# Import package.
from .chain import *
from .comms import *
from .compress import *
from .dataset import *
from .r2_dataset import *
from .hparams import *
from .logging import *
from .schemas import *
from .shard_index import ShardIndex
from .sharded_dataset import ShardedGPUDataset, get_sharded_gpu_dataloader
from .wandb import initialize_wandb
from .neural.demo import DeMo
from .neural.strategies import SimpleAccum, Diloco

# hint type for logger
from logging import Logger
logger: Logger

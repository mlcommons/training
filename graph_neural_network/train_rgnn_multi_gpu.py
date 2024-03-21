import argparse, datetime, os
import numpy as np
import os.path as osp
import sklearn.metrics
import time, tqdm
import torch
import warnings

import torch.distributed as dist
import graphlearn_torch as glt
import mlperf_logging.mllog.constants as mllog_constants

from torch.nn.parallel import DistributedDataParallel

from dataset import IGBHeteroDataset
from mlperf_logging_utils import get_mlperf_logger, submission_info
from utilities import create_ckpt_folder
from rgnn import RGNN

warnings.filterwarnings("ignore")
mllogger = get_mlperf_logger(path=osp.dirname(osp.abspath(__file__)))

def evaluate(model, dataloader, current_device, rank, world_size, epoch_num):
  if rank == 0:
    mllogger.start(
        key=mllog_constants.EVAL_START,
        metadata={mllog_constants.EPOCH_NUM: epoch_num},
    )
  predictions = []
  labels = []
  with torch.no_grad():
    for batch in dataloader:
      batch_size = batch['paper'].batch_size
      out = model(
        {
          node_name: node_feat.to(current_device).to(torch.float32)
          for node_name, node_feat in batch.x_dict.items()
        },  
        batch.edge_index_dict
      )[:batch_size]
      labels.append(batch['paper'].y[:batch_size].cpu().numpy())
      predictions.append(out.argmax(1).cpu().numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    acc = sklearn.metrics.accuracy_score(labels, predictions)

    torch.cuda.synchronize()
    dist.barrier()

    acc_tensor = torch.tensor(acc).to(current_device)
    torch.distributed.all_reduce(acc_tensor, op=torch.distributed.ReduceOp.SUM)    
    global_acc = acc_tensor.item() / world_size
    if rank == 0:
      mllogger.event(
          key=mllog_constants.EVAL_ACCURACY,
          value=global_acc,
          metadata={mllog_constants.EPOCH_NUM: epoch_num},
      )
      mllogger.end(
          key=mllog_constants.EVAL_STOP,
          metadata={mllog_constants.EPOCH_NUM: epoch_num},
      )
    return acc, global_acc

def run_training_proc(rank, world_size,
    hidden_channels, num_classes, num_layers, model_type, num_heads, fan_out,
    epochs, train_batch_size, val_batch_size, learning_rate, random_seed, dataset, 
    train_idx, val_idx, with_gpu, validation_acc, validation_frac_within_epoch,
    evaluate_on_epoch_end, checkpoint_on_epoch_end, ckpt_steps, ckpt_path):
  if rank == 0:
    if ckpt_steps > 0:
      ckpt_dir = create_ckpt_folder(base_dir=osp.dirname(osp.abspath(__file__)))
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group('nccl', rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)
  glt.utils.common.seed_everything(random_seed)
  current_device =torch.device(rank)
  
  print(f'Rank {rank} init graphlearn_torch NeighborLoader...')
  # Create rank neighbor loader for training
  train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
  train_loader = glt.loader.NeighborLoader(
    data=dataset,
    num_neighbors=[int(fanout) for fanout in fan_out.split(',')],
    input_nodes=('paper', train_idx),
    batch_size=train_batch_size,
    shuffle=True,
    drop_last=False,
    device=current_device,
    seed=random_seed
  )

  # Create rank neighbor loader for validation.
  val_idx = val_idx.split(val_idx.size(0) // world_size)[rank]
  val_loader = glt.loader.NeighborLoader(
    data=dataset,
    num_neighbors=[int(fanout) for fanout in fan_out.split(',')],
    input_nodes=('paper', val_idx),
    batch_size=val_batch_size,
    shuffle=True,
    drop_last=False,
    device=current_device,
    seed=random_seed
  )
  # Load checkpoint
  ckpt = None
  if ckpt_path is not None:
    try:
      ckpt = torch.load(ckpt_path)
    except FileNotFoundError as e:
      print(f"Checkpoint file not found: {e}")
      return -1
    
  # Define model and optimizer.
  model = RGNN(dataset.get_edge_types(),
               dataset.node_features['paper'].shape[1],
               hidden_channels,
               num_classes,
               num_layers=num_layers,
               dropout=0.2,
               model=model_type,
               heads=num_heads,
               node_type='paper').to(current_device)
  if ckpt is not None:
    model.load_state_dict(ckpt['model_state_dict'])
  model = DistributedDataParallel(model,
                                  device_ids=[current_device.index] if with_gpu else None,
                                  find_unused_parameters=True)

  param_size = 0
  for param in model.parameters():
    param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))

  loss_fcn = torch.nn.CrossEntropyLoss().to(current_device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  if ckpt is not None:
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

  batch_num = (len(train_idx) + train_batch_size - 1) // train_batch_size
  validation_freq = int(batch_num * validation_frac_within_epoch)
  is_success = False
  epoch_num = 0

  training_start = time.time()
  for epoch in tqdm.tqdm(range(epochs)):
    model.train()
    total_loss = 0
    train_acc = 0
    idx = 0
    gpu_mem_alloc = 0
    epoch_start = time.time()
    for batch in train_loader:
      idx += 1
      batch_size = batch['paper'].batch_size
      out = model(
        {
          node_name: node_feat.to(current_device).to(torch.float32)
          for node_name, node_feat in batch.x_dict.items()
        },  
        batch.edge_index_dict
      )[:batch_size]
      y = batch['paper'].y[:batch_size]
      loss = loss_fcn(out, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      train_acc += sklearn.metrics.accuracy_score(y.cpu().numpy(),
          out.argmax(1).detach().cpu().numpy())*100
      gpu_mem_alloc += (
          torch.cuda.max_memory_allocated() / 1000000
          if with_gpu
          else 0
      )
      #checkpoint
      if ckpt_steps > 0 and idx % ckpt_steps == 0:
        if with_gpu:
          torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
          epoch_num = round((epoch + idx / batch_num), 2)
          glt.utils.common.save_ckpt(idx + epoch * batch_num, 
                                    ckpt_dir, model.module, optimizer, epoch_num)
        dist.barrier()
      # evaluate
      if idx % validation_freq == 0:
        if with_gpu:
          torch.cuda.synchronize()
        dist.barrier()
        epoch_num = round((epoch + idx / batch_num), 2)
        model.eval()
        rank_val_acc, global_acc = evaluate(model, val_loader, current_device, 
                                 rank, world_size, epoch_num)
        if validation_acc is not None and global_acc >= validation_acc:
          is_success = True
          break
        model.train()

    if with_gpu:
      torch.cuda.synchronize()
    dist.barrier()

    #checkpoint at the end of epoch
    if checkpoint_on_epoch_end:
      if rank == 0:
        epoch_num = epoch + 1
        glt.utils.common.save_ckpt(idx + epoch * batch_num, 
                                   ckpt_dir, model.module, optimizer, epoch_num)
      dist.barrier()

    # evaluate at the end of epoch
    if  evaluate_on_epoch_end and not is_success:
      epoch_num = epoch + 1
      model.eval()
      rank_val_acc, global_acc = evaluate(model, val_loader, current_device, 
                                          rank, world_size, epoch_num)
      if validation_acc is not None and global_acc >= validation_acc:
        is_success = True
      
      #tqdm
      train_acc /= idx
      gpu_mem_alloc /= idx
      tqdm.tqdm.write(
          "Rank{:02d} | Epoch {:03d} | Loss {:.4f} | Train Acc {:.2f} | Val Acc {:.2f} | Time {} | GPU {:.1f} MB".format(
              rank,
              epoch,
              total_loss,
              train_acc,
              rank_val_acc*100,
              str(datetime.timedelta(seconds = int(time.time() - epoch_start))),
              gpu_mem_alloc
          )
      )

    # stop training if success
    if is_success:
      break
  
  #log run status
  if rank == 0:
    status = mllog_constants.SUCCESS if is_success else mllog_constants.ABORTED
    mllogger.end(key=mllog_constants.RUN_STOP,
                 metadata={mllog_constants.STATUS: status,
                           mllog_constants.EPOCH_NUM: epoch_num,
                 }
    )
  print("Total time taken " + str(datetime.timedelta(seconds = int(time.time() - training_start))))


if __name__ == '__main__':
  mllogger.event(key=mllog_constants.CACHE_CLEAR, value=True)
  mllogger.start(key=mllog_constants.INIT_START)

  parser = argparse.ArgumentParser()
  root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'data', 'igbh')
  glt.utils.ensure_dir(root)
  parser.add_argument('--path', type=str, default=root,
      help='path containing the datasets')
  parser.add_argument('--dataset_size', type=str, default='full',
      choices=['tiny', 'small', 'medium', 'large', 'full'],
      help='size of the datasets')
  parser.add_argument('--num_classes', type=int, default=2983,
      choices=[19, 2983], help='number of classes')
  parser.add_argument('--in_memory', type=int, default=1,
      choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
  # Model
  parser.add_argument('--model', type=str, default='rgat',
                      choices=['rgat', 'rsage'])
  # Model parameters
  parser.add_argument('--fan_out', type=str, default='15,10,5')
  parser.add_argument('--train_batch_size', type=int, default=1024)
  parser.add_argument('--val_batch_size', type=int, default=1024)
  parser.add_argument('--hidden_channels', type=int, default=512)
  parser.add_argument('--learning_rate', type=float, default=0.001)
  parser.add_argument('--epochs', type=int, default=2)
  parser.add_argument('--num_layers', type=int, default=3)
  parser.add_argument('--num_heads', type=int, default=4)
  parser.add_argument('--random_seed', type=int, default=42)
  parser.add_argument("--cpu_mode", action="store_true",
      help="Only use CPU for sampling and training, default is False.")
  parser.add_argument("--edge_dir", type=str, default='in')
  parser.add_argument('--layout', type=str, default='COO',
      help="Layout of input graph. Default is COO.")
  parser.add_argument("--pin_feature", action="store_true",
      help="Pin the feature in host memory. Default is False.")
  parser.add_argument("--use_fp16", action="store_true", 
      help="To use FP16 for loading the features. Default is False.")
  parser.add_argument("--validation_frac_within_epoch", type=float, default=0.05,
      help="Fraction of the epoch after which validation should be performed.")
  parser.add_argument("--validation_acc", type=float, default=0.72,
      help="Validation accuracy threshold to stop training once reached.")
  parser.add_argument("--evaluate_on_epoch_end", action="store_true",
        help="Evaluate using validation set on each epoch end.")
  parser.add_argument("--checkpoint_on_epoch_end", action="store_true",
      help="Save checkpoint on each epoch end.")
  parser.add_argument('--ckpt_steps', type=int, default=-1, 
        help="Save checkpoint every n steps. Default is -1, which means no checkpoint is saved.")
  parser.add_argument('--ckpt_path', type=str, default=None, 
        help="Path to load checkpoint from. Default is None.")
  args = parser.parse_args()
  args.with_gpu = (not args.cpu_mode) and torch.cuda.is_available()
  assert args.layout in ['COO', 'CSC', 'CSR']

  glt.utils.common.seed_everything(args.random_seed)
  world_size = torch.cuda.device_count()
  submission_info(mllogger, 'GNN', 'reference_implementation')
  mllogger.event(key=mllog_constants.GLOBAL_BATCH_SIZE, value=world_size*args.train_batch_size)
  mllogger.event(key=mllog_constants.GRADIENT_ACCUMULATION_STEPS, value=1)
  mllogger.event(key=mllog_constants.OPT_NAME, value='adam')
  mllogger.event(key=mllog_constants.OPT_BASE_LR, value=args.learning_rate)
  mllogger.event(key=mllog_constants.SEED,value=args.random_seed)
  mllogger.end(key=mllog_constants.INIT_STOP)
  mllogger.start(key=mllog_constants.RUN_START)

  igbh_dataset = IGBHeteroDataset(args.path, args.dataset_size, args.in_memory,
                                  args.num_classes==2983, True, args.layout, 
                                  args.use_fp16)
  # init graphlearn_torch Dataset.
  glt_dataset = glt.data.Dataset(edge_dir=args.edge_dir)

  glt_dataset.init_node_features(
    node_feature_data=igbh_dataset.feat_dict,
    with_gpu=args.with_gpu and args.pin_feature
  )

  glt_dataset.init_graph(
    edge_index=igbh_dataset.edge_dict,
    layout = args.layout,
    graph_mode='ZERO_COPY' if args.with_gpu else 'CPU',
  )
  
  glt_dataset.init_node_labels(node_label_data={'paper': igbh_dataset.label})

  train_idx = igbh_dataset.train_idx.clone().share_memory_()
  val_idx = igbh_dataset.val_idx.clone().share_memory_()

  print('--- Launching training processes ...\n')
  torch.multiprocessing.spawn(
    run_training_proc,
    args=(world_size, args.hidden_channels, args.num_classes, args.num_layers, 
          args.model, args.num_heads, args.fan_out, args.epochs, 
          args.train_batch_size, args.val_batch_size,
          args.learning_rate, args.random_seed,
          glt_dataset, train_idx, val_idx, args.with_gpu,
          args.validation_acc, args.validation_frac_within_epoch, 
          args.evaluate_on_epoch_end, args.checkpoint_on_epoch_end, 
          args.ckpt_steps, args.ckpt_path),
    nprocs=world_size,
    join=True
  )

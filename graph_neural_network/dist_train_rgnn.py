import argparse, datetime
import os.path as osp
import time, tqdm

import graphlearn_torch as glt
import mlperf_logging.mllog.constants as mllog_constants
import numpy as np
import sklearn.metrics
import torch
import torch.distributed

from mlperf_logging_utils import get_mlperf_logger, submission_info
from torch.nn.parallel import DistributedDataParallel
from utilities import create_ckpt_folder
from rgnn import RGNN

mllogger = get_mlperf_logger(path=osp.dirname(osp.abspath(__file__)))

def evaluate(model, dataloader, current_device, use_fp16, with_gpu, 
             rank, world_size, epoch_num):
  if rank == 0:
    mllogger.start(
        key=mllog_constants.EVAL_START,
        metadata={mllog_constants.EPOCH_NUM: epoch_num},
    )
  predictions = []
  labels = []
  with torch.no_grad():
    for batch in tqdm.tqdm(dataloader):
      batch_size = batch['paper'].batch_size
      if use_fp16:
        x_dict = {node_name: node_feat.to(current_device).to(torch.float32)
                  for node_name, node_feat in batch.x_dict.items()}
      else:
        x_dict = {node_name: node_feat.to(current_device)
                  for node_name, node_feat in batch.x_dict.items()}
      out = model(x_dict,
                  batch.edge_index_dict,
                  num_sampled_nodes_dict=batch.num_sampled_nodes,
                  num_sampled_edges_dict=batch.num_sampled_edges)[:batch_size]
      batch_size = min(out.shape[0], batch_size)
      labels.append(batch['paper'].y[:batch_size].cpu().clone().numpy())
      predictions.append(out.argmax(1).cpu().clone().numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    acc = sklearn.metrics.accuracy_score(labels, predictions)

    if with_gpu:
      torch.cuda.synchronize()
    torch.distributed.barrier()

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

def run_training_proc(local_proc_rank, num_nodes, node_rank, num_training_procs,
    split_training_sampling, hidden_channels, num_classes, num_layers, 
    model_type, num_heads, fan_out, epochs, train_batch_size, val_batch_size, 
    learning_rate,
    random_seed,
    dataset, train_idx, val_idx,
    train_channel_size,
    val_channel_size,
    master_addr,
    training_pg_master_port,
    train_loader_master_port,
    val_loader_master_port,
    with_gpu, trim_to_layer, use_fp16,
    edge_dir, rpc_timeout,
    validation_acc, validation_frac_within_epoch, evaluate_on_epoch_end, 
    checkpoint_on_epoch_end, ckpt_steps, ckpt_path):

  world_size=num_nodes*num_training_procs
  rank=node_rank*num_training_procs+local_proc_rank
  if rank == 0:
    if ckpt_steps > 0:
      ckpt_dir = create_ckpt_folder(base_dir=osp.dirname(osp.abspath(__file__)))
  
  glt.utils.common.seed_everything(random_seed)

  # Initialize graphlearn_torch distributed worker group context.
  glt.distributed.init_worker_group(
    world_size=world_size,
    rank=rank,
    group_name='distributed-igbh-trainer'
  )

  current_ctx = glt.distributed.get_context()
  if with_gpu:
    if split_training_sampling:
      current_device = torch.device((local_proc_rank * 2) % torch.cuda.device_count())
      sampling_device = torch.device((local_proc_rank * 2 + 1) % torch.cuda.device_count())
    else:
      current_device = torch.device(local_proc_rank % torch.cuda.device_count())
      sampling_device = current_device
  else:
    current_device = torch.device('cpu')
    sampling_device = current_device

  # Initialize training process group of PyTorch.
  torch.distributed.init_process_group(
    backend='nccl' if with_gpu else 'gloo',
    timeout=datetime.timedelta(seconds=rpc_timeout),
    rank=current_ctx.rank,
    world_size=current_ctx.world_size,
    init_method='tcp://{}:{}'.format(master_addr, training_pg_master_port)
  )

  # Create distributed neighbor loader for training
  train_idx = train_idx.split(train_idx.size(0) // num_training_procs)[local_proc_rank]
  train_loader = glt.distributed.DistNeighborLoader(
    data=dataset,
    num_neighbors=[int(fanout) for fanout in fan_out.split(',')],
    input_nodes=('paper', train_idx),
    batch_size=train_batch_size,
    shuffle=True,
    drop_last=False,
    edge_dir=edge_dir,
    collect_features=True,
    to_device=current_device,
    random_seed=random_seed,
    worker_options = glt.distributed.MpDistSamplingWorkerOptions(
      num_workers=1,
      worker_devices=sampling_device,
      worker_concurrency=4,
      master_addr=master_addr,
      master_port=train_loader_master_port,
      channel_size=train_channel_size,
      pin_memory=True,
      rpc_timeout=rpc_timeout,
      num_rpc_threads=2
    )
  )
  # Create distributed neighbor loader for validation.
  val_idx = val_idx.split(val_idx.size(0) // num_training_procs)[local_proc_rank]
  val_loader = glt.distributed.DistNeighborLoader(
    data=dataset,
    num_neighbors=[int(fanout) for fanout in fan_out.split(',')],
    input_nodes=('paper', val_idx),
    batch_size=val_batch_size,
    shuffle=True,
    drop_last=False,
    edge_dir=edge_dir,
    collect_features=True,
    to_device=current_device,
    random_seed=random_seed,
    worker_options = glt.distributed.MpDistSamplingWorkerOptions(
      num_workers=1,
      worker_devices=sampling_device,
      worker_concurrency=4,
      master_addr=master_addr,
      master_port=val_loader_master_port,
      channel_size=val_channel_size,
      pin_memory=True,
      rpc_timeout=rpc_timeout,
      num_rpc_threads=2
    )
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
  if with_gpu:
    torch.cuda.set_device(current_device)
  model = RGNN(dataset.get_edge_types(),
               dataset.node_features['paper'].shape[1],
               hidden_channels,
               num_classes,
               num_layers=num_layers,
               dropout=0.2,
               model=model_type,
               heads=num_heads,
               node_type='paper',
               with_trim=trim_to_layer).to(current_device)
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
  for epoch in range(epochs):
    if rank == 0:
      mllogger.start(
        key=mllog_constants.EPOCH_START,
        metadata={mllog_constants.EPOCH_NUM: epoch},
      )
    model.train()
    total_loss = 0
    train_acc = 0
    idx = 0
    gpu_mem_alloc = 0
    epoch_start = time.time()
    for batch in tqdm.tqdm(train_loader):
      idx += 1
      batch_size = batch['paper'].batch_size
      if use_fp16:
        x_dict = {node_name: node_feat.to(current_device).to(torch.float32)
                  for node_name,node_feat in batch.x_dict.items()}
      else:
        x_dict = {node_name: node_feat.to(current_device)
                  for node_name,node_feat in batch.x_dict.items()}
      out = model(x_dict,
                  batch.edge_index_dict,
                  num_sampled_nodes_dict=batch.num_sampled_nodes,
                  num_sampled_edges_dict=batch.num_sampled_edges)[:batch_size]
      batch_size = min(batch_size, out.shape[0])
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
        torch.distributed.barrier()
        if rank == 0:
          epoch_num = round((epoch + idx / batch_num), 2)
          glt.utils.common.save_ckpt(idx + epoch * batch_num, 
                                     ckpt_dir, model.module, optimizer, epoch_num)
        torch.distributed.barrier()
      # evaluate
      if idx % validation_freq == 0:
        if with_gpu:
          torch.cuda.synchronize()
        torch.distributed.barrier()
        epoch_num = round((epoch + idx / batch_num), 2)
        model.eval()
        rank_val_acc, global_acc = evaluate(model, val_loader, current_device, 
                                            use_fp16, with_gpu, rank, 
                                            world_size, epoch_num)
        if validation_acc is not None and global_acc >= validation_acc:
          is_success = True
          break
        model.train()      
    
    train_acc /= idx
    gpu_mem_alloc /= idx

    if with_gpu:
      torch.cuda.synchronize()
    torch.distributed.barrier()

    if rank == 0:
      mllogger.end(
        key=mllog_constants.EPOCH_STOP,
        metadata={mllog_constants.EPOCH_NUM: epoch},
      )
    
    #checkpoint at the end of epoch
    if checkpoint_on_epoch_end:
      if rank == 0:
        epoch_num = epoch + 1
        glt.utils.common.save_ckpt(idx + epoch * batch_num, 
                                   ckpt_dir, model.module, optimizer, epoch_num)
      torch.distributed.barrier()

    # evaluate at the end of epoch
    if evaluate_on_epoch_end and not is_success:
      epoch_num = epoch + 1
      model.eval()
      rank_val_acc, global_acc = evaluate(model, val_loader, current_device, 
                                          use_fp16, with_gpu, rank, world_size, 
                                          epoch_num)
      if validation_acc is not None and global_acc >= validation_acc:
        is_success = True
    
      tqdm.tqdm.write(
          "Rank{:02d} | Epoch {:03d} | Loss {:.4f} | Train Acc {:.2f} | Val Acc {:.2f} | Time {} | GPU {:.1f} MB".format(
              current_ctx.rank,
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
  parser.add_argument('--in_memory', type=int, default=0,
      choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
  # Model
  parser.add_argument('--model', type=str, default='rgat',
                      choices=['rgat', 'rsage'])
  # Model parameters
  parser.add_argument('--fan_out', type=str, default='15,10,5')
  parser.add_argument('--train_batch_size', type=int, default=512)
  parser.add_argument('--val_batch_size', type=int, default=512)
  parser.add_argument('--hidden_channels', type=int, default=512)
  parser.add_argument('--learning_rate', type=float, default=0.001)
  parser.add_argument('--epochs', type=int, default=2)
  parser.add_argument('--num_layers', type=int, default=3)
  parser.add_argument('--num_heads', type=int, default=4)
  parser.add_argument('--random_seed', type=int, default=42)
  # Distributed settings.
  parser.add_argument("--num_nodes", type=int, default=2,
      help="Number of distributed nodes.")
  parser.add_argument("--node_rank", type=int, default=0,
      help="The current node rank.")
  parser.add_argument("--num_training_procs", type=int, default=2,
      help="The number of traning processes per node.")
  parser.add_argument("--master_addr", type=str, default='localhost',
      help="The master address for RPC initialization.")
  parser.add_argument("--training_pg_master_port", type=int, default=12111,
      help="The port used for PyTorch's process group initialization across training processes.")
  parser.add_argument("--train_loader_master_port", type=int, default=12112,
      help="The port used for RPC initialization across all sampling workers of train loader.")
  parser.add_argument("--val_loader_master_port", type=int, default=12113,
      help="The port used for RPC initialization across all sampling workers of val loader.")
  parser.add_argument("--cpu_mode", action="store_true",
      help="Only use CPU for sampling and training, default is False.")
  parser.add_argument("--edge_dir", type=str, default='out',
      help="sampling direction, can be 'in' for 'by_dst' or 'out' for 'by_src' for partitions.")
  parser.add_argument('--layout', type=str, default='COO',
      help="Layout of input graph: CSC, CSR, COO. Default is COO.")
  parser.add_argument('--train_channel_size', type=str, default='16GB',
      help="Size of shared memory queue to put sampled results for train dataset")
  parser.add_argument('--val_channel_size', type=str, default='16GB',
      help="Size of shared memory queue to put sampled results for val dataset")
  parser.add_argument("--rpc_timeout", type=int, default=180,
      help="rpc timeout in seconds")
  parser.add_argument("--split_training_sampling", action="store_true",
      help="Use seperate GPUs for training and sampling processes.")
  parser.add_argument("--with_trim", action="store_true",
      help="use trim_to_layer function from PyG")
  parser.add_argument("--use_fp16", action="store_true",
      help="load node/edge feature using fp16 format to reduce memory usage")
  parser.add_argument("--validation_frac_within_epoch", type=float, default=0.05,
      help="Fraction of the epoch after which validation should be performed.")
  parser.add_argument("--validation_acc", type=float, default=0.72,
      help="Validation accuracy threshold to stop training once reached.")
  parser.add_argument("--evaluate_on_epoch_end", action="store_true",
      help="Evaluate using validation set on each epoch end."),
  parser.add_argument("--checkpoint_on_epoch_end", action="store_true",
    help="Save checkpoint on each epoch end."),
  parser.add_argument('--ckpt_steps', type=int, default=-1, 
      help="Save checkpoint every n steps. Default is -1, which means no checkpoint is saved.")
  parser.add_argument('--ckpt_path', type=str, default=None, 
      help="Path to load checkpoint from. Default is None.")
  args = parser.parse_args()
  assert args.layout in ['COO', 'CSC', 'CSR']

  glt.utils.common.seed_everything(args.random_seed)  
  # when set --cpu_mode or GPU is not available, use cpu only mode.
  args.with_gpu = (not args.cpu_mode) and torch.cuda.is_available()
  if args.with_gpu:
    assert(not args.num_training_procs > torch.cuda.device_count())
    if args.split_training_sampling:
      assert(not args.num_training_procs > torch.cuda.device_count() // 2)
  
  world_size = args.num_nodes * args.num_training_procs
  if args.node_rank == 0:
    submission_info(mllogger, mllog_constants.GNN, 'reference_implementation')
    mllogger.event(key=mllog_constants.GLOBAL_BATCH_SIZE, value=world_size*args.train_batch_size)
    mllogger.event(key=mllog_constants.GRADIENT_ACCUMULATION_STEPS, value=1)
    mllogger.event(key=mllog_constants.OPT_NAME, value='adam')
    mllogger.event(key=mllog_constants.OPT_BASE_LR, value=args.learning_rate)
    mllogger.event(key=mllog_constants.SEED,value=args.random_seed)
    mllogger.end(key=mllog_constants.INIT_STOP)
    mllogger.start(key=mllog_constants.RUN_START)

  print('--- Loading data partition ...\n')
  data_pidx = args.node_rank % args.num_nodes
  dataset = glt.distributed.DistDataset(edge_dir=args.edge_dir)
  dataset.load(
    root_dir=osp.join(args.path, f'{args.dataset_size}-partitions'),
    partition_idx=data_pidx,
    graph_mode='ZERO_COPY' if args.with_gpu else 'CPU',
    input_layout = args.layout,
    feature_with_gpu=args.with_gpu,
    whole_node_label_file={'paper': osp.join(args.path, f'{args.dataset_size}-label', 'label.pt')}
  )
  train_idx = torch.load(
    osp.join(args.path, f'{args.dataset_size}-train-partitions', f'partition{data_pidx}.pt')
  )
  val_idx = torch.load(
    osp.join(args.path, f'{args.dataset_size}-val-partitions', f'partition{data_pidx}.pt')
  )
  train_idx.share_memory_()
  val_idx.share_memory_()

  if args.node_rank == 0:
    mllogger.event(key=mllog_constants.TRAIN_SAMPLES, value=train_idx.size(0) * world_size)
    mllogger.event(key=mllog_constants.EVAL_SAMPLES, value=val_idx.size(0) * world_size)
  
  print('--- Launching training processes ...\n')
  torch.multiprocessing.spawn(
    run_training_proc,
    args=(args.num_nodes, args.node_rank, args.num_training_procs, 
          args.split_training_sampling, args.hidden_channels, args.num_classes, 
          args.num_layers, args.model, args.num_heads, args.fan_out, 
          args.epochs, args.train_batch_size, args.val_batch_size, args.learning_rate,
          args.random_seed,
          dataset, train_idx, val_idx,
          args.train_channel_size,
          args.val_channel_size,
          args.master_addr,
          args.training_pg_master_port,
          args.train_loader_master_port,
          args.val_loader_master_port,
          args.with_gpu,
          args.with_trim,
          args.use_fp16,
          args.edge_dir,
          args.rpc_timeout,
          args.validation_acc, 
          args.validation_frac_within_epoch,
          args.evaluate_on_epoch_end,
          args.checkpoint_on_epoch_end,
          args.ckpt_steps,
          args.ckpt_path),
    nprocs=args.num_training_procs,
    join=True
  )

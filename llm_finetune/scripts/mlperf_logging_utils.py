import os
from mlperf_logging import mllog
from mlperf_logging.mllog import constants
import torch
import torch.distributed as dist
from transformers import  TrainingArguments,  TrainerCallback, TrainerState, TrainerControl



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def barrier():
    if not is_dist_avail_and_initialized():
        return
    torch.distributed.barrier()

class LoraLogger:
    def __init__(self,target_eval_loss=None, filename=None, default_stack_offset=2):
        self.mllogger = mllog.get_mllogger()
        mllog.config(default_stack_offset=default_stack_offset,
                     filename=(filename or os.getenv("COMPLIANCE_FILE") or "mlperf_compliance.log"),
                     root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__))))
        self.target_eval_loss=target_eval_loss

    @property
    def rank(self):
        return get_rank()

    def event(self, key, value=None, metadata=None, sync=False, log_rank=None):
        log_rank = self.rank==0 if log_rank is None else log_rank
        if sync:
            barrier()
        if log_rank:
            self.mllogger.event(key=key, value=value, metadata=metadata)

    def start(self, key, value=None, metadata=None, sync=False, log_rank=None):
        log_rank = self.rank==0 if log_rank is None else log_rank
        if sync:
            barrier()
        if log_rank:
            self.mllogger.start(key=key, value=value, metadata=metadata)

    def end(self, key, value=None, metadata=None, sync=False, log_rank=None):
        log_rank = self.rank==0 if log_rank is None else log_rank
        if sync:
            barrier()
        if log_rank:
            self.mllogger.end(key=key, value=value, metadata=metadata)

def submission_info(mllogger: LoraLogger,
                    submission_benchmark: str, submission_division: str, submission_org: str, submission_platform: str,
                    submission_poc_name: str, submission_poc_email: str, submission_status: str):
    """Logs required for a valid MLPerf submission."""
    if mllogger.rank==0:
        mllogger.event(key=constants.SUBMISSION_BENCHMARK, value=submission_benchmark)
        mllogger.event(key=constants.SUBMISSION_DIVISION, value=submission_division)
        mllogger.event(key=constants.SUBMISSION_ORG, value=submission_org)
        mllogger.event(key=constants.SUBMISSION_PLATFORM, value=submission_platform)
        mllogger.event(key=constants.SUBMISSION_POC_NAME, value=submission_poc_name)
        mllogger.event(key=constants.SUBMISSION_POC_EMAIL, value=submission_poc_email)
        mllogger.event(key=constants.SUBMISSION_STATUS, value=submission_status)

def general_info(mllogger: LoraLogger,args,world_size,eval_samples,train_samples):
    if mllogger.rank==0:
        mllogger.event(key=constants.GLOBAL_BATCH_SIZE, value=args.per_device_train_batch_size*args.gradient_accumulation_steps*world_size)
        mllogger.event(key=constants.TRAIN_SAMPLES, value=train_samples)
        mllogger.event(key=constants.EVAL_SAMPLES, value=eval_samples)
        mllogger.event(key=constants.SEED, value=args.seed)

def optimization_info(mllogger: LoraLogger,args):
    """Logs required for a valid MLPerf submission."""
    if mllogger.rank==0:
        mllogger.event(key=constants.OPT_LR_WARMUP_FACTOR, value=args.warmup_ratio)
        mllogger.event(key=constants.OPT_LR_TRAINING_STEPS, value=args.max_steps)
        mllogger.event(key=constants.OPT_BASE_LR, value=args.learning_rate)
        #mllogger.event(key=constants.OPT_LAMB_BETA_1, value=args.beta1)

class MLPerfCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    def __init__(self,logger):
        super().__init__()
        self.mllogger = logger
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.mllogger.start(constants.RUN_START,value='')

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if state.global_step % (state.logging_steps) == 0 and state.global_step > 0 and not state.global_step % (state.eval_steps) == 0:
            self.mllogger.event('train_loss',value=state.log_history[-1]['loss'],metadata={"steps":state.log_history[-1]['step']})
            control.should_log = True
            
        if state.global_step % (state.eval_steps) == 0 and state.global_step>0:
            self.mllogger.event('eval_loss',value=state.log_history[-1]['eval_loss'],metadata={"steps":state.log_history[-1]['step']})
            control.should_log = True
            
        eval_loss_list=[sl['eval_loss'] for sl in state.log_history if 'eval_loss' in sl]
        if eval_loss_list and eval_loss_list[-1]<=self.mllogger.target_eval_loss: 
            control.should_training_stop = True
            self.mllogger.end(constants.RUN_STOP,value=eval_loss_list[-1],metadata={"steps":state.log_history[-1]['step'],"status": 'success'})
        if state.global_step >= state.max_steps:
            control.should_training_stop = True
            self.mllogger.end(constants.RUN_STOP,value=eval_loss_list[-1],metadata={"steps":state.log_history[-1]['step'],"status": 'fail'})
        
        return control
    

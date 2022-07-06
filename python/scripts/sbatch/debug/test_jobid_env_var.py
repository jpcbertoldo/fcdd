print('in `test_jobid_env_var.py`')
import os
print(f"type(os.environ) = {type(os.environ)}")

for k in sorted(os.environ.keys()):
    if not k.lower().startswith('slurm'):
        continue
    print(f"{k} = {os.environ[k]}")
    
print(os.environ.get('ashsahsahshashah'))
# os.environ['ashsahsahshashah']


print("SLURM_CLUSTER_NAME".lower() + '=os.environ.get("SLURM_CLUSTER_NAME")')
print("SLURMD_NODENAME".lower() + '=os.environ.get("SLURMD_NODENAME")')
print("SLURM_JOB_PARTITION".lower() + '=os.environ.get("SLURM_JOB_PARTITION")')
print("SLURM_SUBMIT_HOST".lower() + '=os.environ.get("SLURM_SUBMIT_HOST")')
print("SLURM_JOB_USER".lower() + '=os.environ.get("SLURM_JOB_USER")')
print("SLURM_TASK_PID".lower() + '=os.environ.get("SLURM_TASK_PID")')
print("SLURM_JOB_NAME".lower() + '=os.environ.get("SLURM_JOB_NAME")')
print("SLURM_ARRAY_JOB_ID".lower() + '=os.environ.get("SLURM_ARRAY_JOB_ID")')
print("SLURM_ARRAY_TASK_ID".lower() + '=os.environ.get("SLURM_ARRAY_TASK_ID")')
print("SLURM_JOB_ID".lower() + '=os.environ.get("SLURM_JOB_ID")')


print(f'os.environ.get("SLURM_STEP_GPUS")={os.environ.get("SLURM_STEP_GPUS")}')
print(f'os.environ.get("SLURM_JOB_GPUS")={os.environ.get("SLURM_JOB_GPUS")}')
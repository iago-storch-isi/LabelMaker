from clearml import Task, Model, Dataset, OutputModel
from clearml.backend_api.session.client import APIClient

# EXCEPTIONS
class UnexistentTaskModel(RuntimeError):
    pass

class InvalidClearmlID(RuntimeError):
    pass

# FUNCTIONS
def get_clearml_object_type(id_: str):
    api = APIClient()
    
    try:
        # If no errors detected, it is a task
        api.tasks.get_by_id(task=id_)
        return Task
    except Exception:
        pass
    try:
        # If no errors detected, it is a model
        api.models.get_by_id(id_)
        return Model
    except Exception:
        pass
    return None

def get_last_model(task: Task):
        return task.get_models()['output'][-1]

def set_task_output_model(task: Task, model_path: str, target_name: str = "model_weights.pth"):
    """
    Given a task and a .pth model filepath, defines the model as
    the OutputModel of the task
    """
    task_output_model = OutputModel(
        task,
        framework="PyTorch",
        name=f"model_{task.name}",
    )
    task_output_model.update_weights(
        weights_filename=model_path,
        target_filename=target_name,
        auto_delete_file=False,
    )
    task_output_model.set_upload_destination(
        task._get_default_report_storage_uri()
    )
    return task_output_model

import clearml
import re
import clearml.datasets
import subprocess
import warnings
from pathlib import Path
import shutil
import os
import yaml
from munch import Munch
import argparse


class UnpushedCommitException(RuntimeError):
    pass


def get_args():
    parser = argparse.ArgumentParser(description="Run script on remote clearml worker")

    defaults = dict(
        project="teste_iago/teste",
        task="ARKitLabelMaker run - {command}",
        docker="10.167.1.54/apt/arkit-labelmaker:v0.1",
        docker_arguments="--shm-size=64000mb -e MKL_SERVICE_FORCE_INTEL=1",
        docker_setup_bash_script=[
            "echo '[!] INITIALIZE SETUP BASH SCRIPT'",
            "export 'CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/miniconda3/bin/python'",
            "export PATH='/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin/bin:$PATH'",
            "echo PATH && echo $PATH",
            "echo PYTHONPATH && echo $PYTHONPATH",
            "echo PYTHONHOME && echo $PYTHONHOME",
            "python --version",
            "conda --version",
            "echo '[!] FINISH SETUP BASH SCRIPT'",
        ],
        local=False,
        config="",
        queue="arkit",
    )

    parser.add_argument(
        "command",
        help="command-line command to run on remote worker (use quotes to avoid confusing parameters with this script, e.g. 'bash remote.sh -g 2 -d scannet')",
    )
    parser.add_argument(
        "--queue", help="remote clearML queue to run on", default=defaults["queue"]
    )
    parser.add_argument(
        "--project", help="name of clearml project", default=defaults["project"]
    )
    parser.add_argument("--task", help="name of clearml task", default=defaults["task"])
    parser.add_argument(
        "--docker",
        help="complete tag of docker image to use on remote",
        default=defaults["docker"],
    )
    parser.add_argument(
        "--docker_arguments",
        help="additional arguments to use for `docker run` on remote",
        default=defaults["docker_arguments"],
    )
    parser.add_argument(
        "--docker_setup_bash_script",
        help="bash script to run at the beginning of the docker before launching the Task itself",
        default=defaults["docker_setup_bash_script"],
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="don't run remotely, good for testing locally",
        default=defaults["local"],
    )
    parser.add_argument(
        "--config",
        help="YAML config file with additional arguments to specific runs (e.g. model training)",
        default=defaults["config"],
    )
    parser.add_argument(
        "--update-config",
        nargs="+",
        help="Substitute config params in the YAML config file in the format param1=a param2=b",
    )

    # Parser args and substitute placeholders in the format {value} to the value itself (see argument --task).
    args = parser.parse_args()
    for arg in args.__dict__:
        value = getattr(args, arg)
        if isinstance(value, str) and re.search("{\w+}", value):
            setattr(args, arg, value.format(**args.__dict__))

    return args


def load_config(cfg_path, update_config):
    cfg_txt = open(cfg_path, "r").read()
    # parse as Munch obj
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    # Update config based on args
    if update_config:
        for param in update_config:
            key, value = param.split("=")
            cfg[key] = value
        print("Configuration updated:", cfg)
    return cfg


def download_dataset(cfg):
    # Download dataset to `args.dataset_path`
    print("Downloading dataset")
    data_path = clearml.datasets.Dataset.get(dataset_id=cfg.dataset_id).get_local_copy()
    dataset_path = Path(cfg.dataset_path)
    if not dataset_path.exists():
        dataset_path.parent.mkdir(exist_ok=True, parents=True)
    elif os.path.islink(dataset_path):
        os.unlink(dataset_path)
    print(f"Symlinking {data_path} to {dataset_path}")
    os.symlink(src=data_path, dst=dataset_path)


def download_model_pretrain(cfg, current_task: clearml.Task):
    # Download model to pretrain_path
    pretrain_id = cfg.pretrain
    pretrain_path = Path(cfg.pretrain_path)
    
    if pretrain_id:
        # Resolve pretrain ID (user can put a model ID or a task ID that generated the model)
        obj_type = get_clearml_object_type(pretrain_id)
        if obj_type is clearml.Model:
            print("Pretrained model ID detected. Downloading model")
            cml_model = clearml.Model(pretrain_id)
            src_path = cml_model.get_local_copy()
        elif obj_type is clearml.Task:
            print("Task ID detected. Checking pretrained models")
            pretrain_task: clearml.Task = clearml.Task.get_task(pretrain_id)
            
            # Check if is a training task
            if pretrain_task.task_type != clearml.TaskTypes.training:
                raise UnexistentTaskModel(
                    f"Task '{pretrain_id}' in 'pretrain' configuration it's not of type {clearml.TaskTypes.training}."
                )
            
            # Download model if the task has it defined in its outputs
            if pretrain_task.get_models()['output']:
                print("Found OutputModel in Task. Downloading model")
                cml_model: clearml.OutputModel = get_last_model(pretrain_task)
                src_path = cml_model.get_local_copy()
            
            # Or download model from registered artifacts
            elif 'model' in pretrain_task.artifacts:
                print("Pretrained model found as normal artifact. Downloading model.")
                src_path = pretrain_task.artifacts['model'].get_local_copy()
                # Set the artifact model as the output model of the previous pretrain task
                print("Setting artifact model as OutputModel")
                cml_model = set_task_output_model(pretrain_task, src_path)
            # The task does not have a model associated
            else:
                raise UnexistentTaskModel(f"No output model was found for the task '{pretrain_id}'.")
        else:
            raise InvalidClearmlID(
                "Invalid pretrain ID. Check if 'pretrain' defined in the configuration is a valid model or task ID."
            )
        
        current_task.set_input_model(cml_model.id, name="pretrained_model")
        
        # Copy the pth model file to the experiment configured pretrain_path
        pretrain_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(src_path, pretrain_path)


def execute_training_command(task: clearml.Task, command, cfg):
    task.set_task_type(clearml.TaskTypes.training)
    download_dataset(cfg)
    download_model_pretrain(cfg, task)

    # Run command on remote
    captured_e = None
    try:
        print("Starting training command")
        subprocess.run(command.split(), stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"WARNING: called command failed {command}")
        captured_e = e
    finally:
        # Tries to upload model to task artifacts
        models = list(Path(".").glob(cfg.output_model_location))
        if len(models) > 0:
            model_path = models[0]
            set_task_output_model(task, model_path)
        else:
            print(
                f"No model found on {cfg.output_model_location}, skipping artifact upload"
            )

    if captured_e:
        task.mark_failed()
        raise captured_e


def execute_inference_command(task, command, cfg):
    task.set_task_type(clearml.TaskTypes.inference)
    download_dataset(cfg)
    download_model_pretrain(cfg, task)

    # Inference can be a generic command, but it has the two dependencies above
    execute_generic_command(task, command)


def execute_evaluation_from_files_command(task, command, cfg):
    task.set_task_type(clearml.TaskTypes.testing)
    # No need for model
    download_dataset(cfg)
    execute_generic_command(task, command)


def execute_model_upload_command(task, command, cfg):
    model_to_be_uploaded = Path(cfg.output_model_location)
    download_model_pretrain(cfg, task)
    execute_generic_command(task, command)
    set_task_output_model(task, str(model_to_be_uploaded))
    print(f"Model {model_to_be_uploaded.name} uploaded sucessfully")


def execute_generic_command(task: clearml.Task, command):
    captured_e = None
    # Run command on remote
    try:
        subprocess.run(command.split(), stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"WARNING: called command failed {command}")
        captured_e = e
    if captured_e:
        task.mark_failed()
        raise captured_e


def main(args):
    # Parse configuration file
    if args.config:
        cfg = load_config(args.config, args.update_config)
    else:
        cfg = None

    # requirements_path = "env_v2/requirements.txt"

    # # Set packages
    # Task.force_requirements_env_freeze(
    #     force=True,
    #     requirements_file=requirements_path,
    # )


    # Setup ClearML task and put into the queue
    task: clearml.Task = clearml.Task.init(
        project_name=args.project, task_name=args.task
    )
    
    
    # task.set_packages(Path(requirements_path).read_text().splitlines())



    os.environ["FORCED_CLEARML_TASK_ID"] = task.id

    task.connect(args)

    task.set_base_docker(
        docker_image=args.docker,
        docker_arguments=args.docker_arguments,
        docker_setup_bash_script=args.docker_setup_bash_script,
    )

    task.set_packages(["."])

    # Execution stops here and continues on remote
    if not args.local:
        task.execute_remotely(queue_name=args.queue)

    # If a config file is used, it executes extra steps depending on the task
    # Current commands:
    # - Generic: No cfg file. Executes a simple command, example: "mkdir test_folder".
    # - ML Training: run_training_config.yaml. Executes a training procedure together with steps
    # of dataset and model download and model saving to the clearml artifacts.
    if cfg:
        if cfg.config_type == "training":
            print("Executing Training Procedure")
            execute_training_command(task, args.command, cfg)
        elif cfg.config_type == "inference":
            print("Executing Inference Procedure")
            execute_inference_command(task, args.command, cfg)
        elif cfg.config_type == "evaluation_from_files":
            print("Executing Evaluation from files")
            execute_evaluation_from_files_command(task, args.command, cfg)
        elif cfg.config_type == "update_local_model":
            print("Executing Model Upload Procedure")
            execute_model_upload_command(task, args.command, cfg)

    else:
        print("Executing Generic Procedure")
        execute_generic_command(task, args.command)


def check_git_status(args):
    not_in_remote = subprocess.getoutput(cmd=f"git log HEAD --not --remotes")
    untracked_files = subprocess.getoutput(
        cmd=f"git ls-files --others --exclude-standard"
    )

    if not args.local:
        if not_in_remote.strip() != "":
            raise UnpushedCommitException(
                "Can't run remotely because there's unpushed commits in HEAD."
                "\nGit log output:\n"
                f"{not_in_remote}"
            )
        elif untracked_files.strip() != "":
            warnings.warn(
                "UNTRACKED FILES! You have untracked files on git,"
                "which might result in errors on the remote:"
                f"\n{untracked_files}"
            )


if __name__ == "__main__":
    args = get_args()
    check_git_status(args)
    main(args)

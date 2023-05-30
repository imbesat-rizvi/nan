import os
import yaml
from pathlib import Path
from argparse import ArgumentParser
import torch
import torch.utils.data as torch_dataset
from datasets import Dataset
import pandas as pd

# import neptune # when neptune is installed
import neptune.new as neptune  # when neptune-client is installed
from neptune import management

try:
    import lightning.pytorch as pl  # latest
except ModuleNotFoundError:
    import pytorch_lightning as pl  # older


from data.probing import gen_reconstruct_set, gen_arith_op_set
from data.utils import get_data_from_LitData
from models.probing import (
    Reconstructor,
    LitReconstructor,
    ArithmeticOperator,
    LitArithmeticOperator,
)

from models.numerical_reasoning import (
    EncoderNRModel,
    LitEncoderNRModel,
)

from models.probing.plot_utils import plot_reconstruction


EXP_DATA = dict(
    reconstruction=gen_reconstruct_set,
    arithmetic_operations=gen_arith_op_set,
    numerical_reasoning=get_data_from_LitData,
)

EXP_MODELS = dict(
    reconstruction=dict(neural_net=Reconstructor, model=LitReconstructor),
    arithmetic_operations=dict(
        neural_net=ArithmeticOperator, model=LitArithmeticOperator
    ),
    numerical_reasoning=dict(neural_net=EncoderNRModel, model=LitEncoderNRModel),
)


def save_predictions_and_target(
    predictions,
    target_or_dataset=None,
    target_cols="labels",
    file_prefix="test",
    save_path="predictions_and_target.csv",
):

    if isinstance(target_cols, str):
        target_cols = [target_cols]

    target = target_or_dataset

    if isinstance(target_or_dataset, torch_dataset.DataLoader):
        target_or_dataset = target_or_dataset.dataset

    if isinstance(target_or_dataset, Dataset):
        target = target_or_dataset.remove_columns(
            [c for c in target_or_dataset.column_names if c not in target_cols]
        )
        target.set_format("numpy")
        target = pd.DataFrame(target)

    elif isinstance(target_or_dataset, torch_dataset.Dataset):
        # Usually 1st entry is model input and 2nd entry is target
        target = torch.cat([torch.atleast_1d(i[1]) for i in target_or_dataset])
        target = pd.DataFrame(target, columns=target_cols)

    if isinstance(predictions, list) and isinstance(predictions[0], torch.Tensor):
        predictions = torch.cat(predictions)

    data_to_save = pd.DataFrame(predictions, columns=[f"{i}_pred" for i in target_cols])
    if target is not None:
        data_to_save = pd.concat((data_to_save, target), axis=1)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path = save_path.parent / f"{file_prefix}-{save_path.name}"
    data_to_save.to_csv(save_path, index=False)
    return save_path


def main(exp_config, exp_data=EXP_DATA, exp_models=EXP_MODELS, run=None):

    train_data, val_data, test_data = exp_data[exp_config["name"]](
        **exp_config["data_args"]
    )

    neural_net = exp_models[exp_config["name"]]["neural_net"](
        **exp_config["neural_net_args"]
    )

    model = exp_models[exp_config["name"]]["model"](
        neural_net, **exp_config["model_args"]
    )

    logger = None
    if run is not None:
        logger = pl.loggers.NeptuneLogger(run=run, log_model_checkpoints=False)

    Path(exp_config["trainer_args"]["default_root_dir"]).mkdir(
        parents=True, exist_ok=True
    )

    trainer = pl.Trainer(
        callbacks=[
            getattr(pl.callbacks, k)(**v)
            for k, v in exp_config["trainer_callbacks"].items()
        ],
        logger=logger,
        **exp_config["trainer_args"],
    )

    trainer.fit(model, train_data, val_data)

    trainer.test(dataloaders=train_data)
    trainer.test(dataloaders=val_data)
    trainer.test(dataloaders=test_data)

    train_pred = trainer.predict(dataloaders=train_data)
    val_pred = trainer.predict(dataloaders=val_data)
    test_pred = trainer.predict(dataloaders=test_data)

    # import pdb; pdb.set_trace()

    if exp_config.get("result_args", {}).get("predictions"):
        pred_target_pairs = {
            "train": (train_pred, train_data),
            "val": (val_pred, val_data),
            "test": (test_pred, test_data),
        }

        for split, (pred, target) in pred_target_pairs.items():
            save_path = save_predictions_and_target(
                predictions=pred,
                target_or_dataset=target,
                file_prefix=split,
                **exp_config["result_args"]["predictions"],
            )

            if run is not None:
                run[f"reference/{save_path.name}"].upload(str(save_path))

    if exp_config["name"] == "reconstruction":
        save_path = (
            Path(exp_config["result_args"]["plot_args"]["save_path"])
            / "reconstruction.jpg"
        )

        plot_reconstruction(
            x=(train_data, val_data, test_data),
            x_pred=(train_pred, val_pred, test_pred),
            interp_range=exp_config["data_args"]["interp_range"],
            save_path=save_path,
        )

        if run is not None:
            run[f"reference/{save_path.name}"].upload(str(save_path))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Arguments for NaN: Numbers are Numbers encoding experiments."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="path to config file specifying parameters of experiments.",
    )

    parser.add_argument(
        "--private",
        default="private.yaml",
        help="path to private file specifying api keys and tokens.",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.private) as f:
        private = yaml.safe_load(f)

    run = None
    if "neptune" in config:
        os.environ["NEPTUNE_API_TOKEN"] = private["neptune-api-token"]
        workspace = config["neptune"]["workspace"]
        project_name = config["neptune"]["project-name"]
        project = f"{workspace}/{project_name}"

        if project not in management.get_project_list():
            project = management.create_project(
                name=project_name,
                workspace=workspace,
                visibility="priv",
            )

        run = neptune.init_run(
            project=project,
            name=config["neptune"]["run-name"],
            tags=config["neptune"]["tags"],
        )

        run["reference/config.yaml"].upload(args.config)

    main(exp_config=config["exp"], run=run)

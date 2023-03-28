import os
import yaml
from argparse import ArgumentParser

# import neptune
import neptune.new as neptune
from neptune import management


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

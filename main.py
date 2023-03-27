import os
import yaml
import neptune
from neptune import management


if __name__ == "__main__":
    with open("private.yaml") as f:
        private = yaml.safe_load(f)

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    if "neptune" in config:
        os.environ["NEPTUNE_API_TOKEN"] = private["neptune-api-token"]
        workspace = config["neptune"]["workspace"]
        project_name = config["neptune"]["project-name"]
        project = f"{workspace}/{project_name}"
        
        print(management.get_project_list())
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
import yaml
import neptune


if __name__ == "__main__":
    with open("private.yaml") as f:
        private = yaml.safe_load(f)
    run = neptune.init_run(
        name="NaN-numeracy-probing-encoding-without-LM",
        project="NaN",
        tags=["numeric-encoding", "numeracy-probing", "without-LM"],
        api_token=private["neptune-api-token"],
    )
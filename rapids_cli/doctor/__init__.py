import yaml
from rapids_cli.doctor.checks import default_checks
from rapids_cli.doctor.checks.cudf import cudf_checks
from rapids_cli.config import config


VALID_SUBCOMMANDS = config['valid_subcommands']['VALID_SUBCOMMANDS']


def doctor_check(arguments): 
    if len(arguments) == 0:
            default_checks()
    else:
        for argument in arguments: 
            if argument not in VALID_SUBCOMMANDS: 
                print(f"Not a valid subcommand - please use one of the following: {str(VALID_SUBCOMMANDS)}")
            if argument == "cudf":
                with open('config.yml', 'r') as file: 
                    config = yaml.safe_load(file)
                cuda_requirement = config['cudf_requirements']['cuda_requirement']
                driver_requirement = config['cudf_requirements']['driver_requirement']
                compute_requirement = config['cudf_requirements']['compute_requirement']

                cudf_checks(cuda_requirement,driver_requirement, compute_requirement)



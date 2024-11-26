import os
import subprocess


def get_current_shell():
    try:
        pid = os.getppid()
        shell = (
            subprocess.check_output(["ps", "-p", str(pid), "-o", "comm="])
            .decode()
            .strip()
        )
        shell = shell.split("/")[-1]
        return shell
    except Exception as e:
        print(f"Error detecting shell: {e}")
        return None


def setup_completion():

    shell = get_current_shell()

    cli_command = "/opt/miniconda3/envs/rapids-env/bin/rapids"

    rc_file = os.path.expanduser(f"~/.{shell}rc")
    autoload_line = ""
    if shell == "zsh":
        autoload_line = "autoload -U compinit && compinit"

    completion_line = f'eval "$(_{cli_command.upper().replace("-", "_")}_COMPLETE={shell}_source {cli_command})"'
    if os.path.exists(rc_file):
        with open(rc_file, "r") as file:
            if completion_line in file.read():
                return
    with open(rc_file, "a") as file:
        file.write(
            f"\n# Added by {cli_command} installer\n{autoload_line}\n {completion_line} \n"
        )

    print(f"[INFO] Added shell completion for {cli_command} to {rc_file}")


if __name__ == "__main__":
    setup_completion()

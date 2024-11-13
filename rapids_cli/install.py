import os


def setup_completion():
    shell = "zsh"
    cli_command = "rapids"
    rc_file = os.path.expanduser(f"~/.{shell}rc")
    completion_line = f'eval "$(_{cli_command.upper().replace("-", "_")}_COMPLETE={shell}_source {cli_command})"'

    if os.path.exists(rc_file):
        with open(rc_file, "r") as file:
            if completion_line in file.read():
                return
    with open(rc_file, "a") as file:
        file.write(f"\n# Added by {cli_command} installer\n{completion_line}\n")

    print(f"[INFO] Added shell completion for {cli_command} to {rc_file}")


if __name__ == "__main__":
    setup_completion()

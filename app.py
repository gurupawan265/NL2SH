import subprocess

def convert_to_shell(user_input):
    user_input = user_input.lower()

    if "list files" in user_input:
        return "ls"
    elif "current directory" in user_input:
        return "pwd"
    elif "show root folders" in user_input:
        return "ls /"
    elif "create folder" in user_input:
        return "mkdir new_folder"
    else:
        return "echo Command not understood"


def run_in_docker(command):
    docker_cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", "256m",
        "--cpus", "1",
        "nl2sh-sandbox",
        "bash", "-c", command
    ]

    result = subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True
    )

    return result.stdout, result.stderr


if __name__ == "__main__":
    user_input = input("Enter natural language command: ")

    shell_command = convert_to_shell(user_input)
    print("Generated shell command:", shell_command)

    output, error = run_in_docker(shell_command)

    if output:
        print("OUTPUT:\n", output)
    if error:
        print("ERROR:\n", error)

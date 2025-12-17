import subprocess


class Sandbox:
    def __init__(self):
        self.container = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "ubuntu:24.04",
                "sleep",
                "infinity"
            ],
            text = True,
            check = True,
            timeout = 30,
            capture_output = True,
        )
        self.container_id = self.container.stdout

    def exec(self, *args, **kwargs):
        args = ["docker", "exec", self.container_id] + args
        return subprocess.run(*args, **kwargs)

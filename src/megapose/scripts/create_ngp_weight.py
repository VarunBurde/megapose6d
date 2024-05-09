import os
import subprocess

def run_cmd(cmd: str):
    """Run a command in the terminal."""
    print("cmd:", cmd)
    print("output:")
    subprocess.Popen(cmd, shell=True).wait()

if __name__ == '__main__':
    path = "/home/testbed/Projects/instant-ngp/data/train_pbr"
    ngp_script = "/home/testbed/Projects/instant-ngp/scripts/run.py"
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            if dir != "000006":
                continue
            print("Processing", dir)
            scene = "--scene " + os.path.join(path, dir, "transforms.json")
            train = "--train"

            os.makedirs(os.path.join(path, dir, "mesh"), exist_ok=True)
            os.makedirs(os.path.join(path, dir, "snapshot"), exist_ok=True)
            save_snapshot = "--save_snapshot " + os.path.join(path, dir, "snapshot", "base.ingp")

            cmd = f"python {ngp_script} {scene} {train} {save_snapshot}"
            run_cmd(cmd)
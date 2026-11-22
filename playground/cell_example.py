import os, subprocess
os.environ["PG_PASS"] = "choose-a-password"
os.environ["PG_MODE"] = "dual"
subprocess.run("rm -rf /kaggle/working/pipe; git clone -q --depth 1 https://github.com/ShiivamSinghChauhan/character-LoRA-training-pipeline /kaggle/working/pipe", shell=True)
exec(open("/kaggle/working/pipe/playground/launch.py").read())

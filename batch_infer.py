import os
import subprocess

input_file = "my_java_methods.txt"
output_file = "pred_completions.txt"
temp_file = "temp_pred.txt"

out_dir = "out-jam-cgpt"
checkpoint = "ckpt_pretrain.pt"
config = "config/finetune_model_350m_dataset_170k.py"

with open(input_file, "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

with open(output_file, "w") as outf:
    for idx, prompt in enumerate(prompts):
        print(f"[{idx+1}/{len(prompts)}] Prompt: {prompt}")
        cmd = [
            "python3", "sample_jam_cgpt.py", config,
            "--out_dir=" + out_dir,
            "--outfilename=" + checkpoint,
            "--prediction_filename=" + temp_file,
            "--start=" + prompt,
            "--num_samples=1",
            "--max_new_tokens=60",
            "--temperature=0.8",
            "--top_k=50"
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(temp_file):
            with open(temp_file, "r") as tf:
                result = tf.read().strip()
            outf.write(f"Prompt: {prompt}\nCompletion:\n{result}\n{'='*60}\n")
            os.remove(temp_file)

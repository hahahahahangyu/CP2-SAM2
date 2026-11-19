# Running the benchmark on the cloud GPU (step by step)

This guide covers the whole run. You copy each grey box into the terminal and press Enter, and nothing else needs editing.

**Time:** about 1 hour, most of it waiting.
**Cost:** about RM 1.30–3 for the run. Remember to stop the machine at the end (Step 10).

---

## Before you start (on your Mac, 2 minutes)

You need two zip files.

1. **Pothole Mix test images.** This file is already made for you:
   `Documents/fyp datasets/SHREC2022/Pothole-Mix/potholemix_test_for_cloud.zip` (57 MB)
2. **NPD.**
   1. In Finder, go to `Downloads`.
   2. Right-click the `NPD` folder and choose **Compress "NPD"**.
   3. You now have `Downloads/NPD.zip`.

---

## Step 1: Start a GPU machine

1. Log in to the GPU platform.
2. Choose the **cheaper** of the two GPU options: **NVIDIA L4** or **NVIDIA T4**. Both are big enough.
3. Launch it with **VS Code**.
4. Wait until VS Code opens in your browser.

## Step 2: Open a terminal

In VS Code's top menu, click **Terminal → New Terminal**. A panel opens at the bottom. You type every command below into that panel.

## Step 3: Download the project code

```bash
git clone https://github.com/hahahahahangyu/CP2-SAM2.git
cd CP2-SAM2
mkdir -p data/raw
```

If it asks you to sign in to GitHub, sign in.

## Step 4: Upload the two zip files

1. In VS Code's left sidebar (Explorer), open `CP2-SAM2` → `data` → `raw`.
2. Drag `potholemix_test_for_cloud.zip` and `NPD.zip` from Finder onto the `raw` folder.
3. Wait until both uploads finish.

## Step 5: Unzip them

```bash
cd data/raw
unzip -q potholemix_test_for_cloud.zip
unzip -q NPD.zip
ls
cd ../..
```

`ls` should list `pothole-mix` and `NPD`, plus the zips and maybe a `__MACOSX` folder, which you can ignore.

If it says `unzip: command not found`, run these two lines instead:

```bash
python3 -m zipfile -e potholemix_test_for_cloud.zip .
python3 -m zipfile -e NPD.zip .
```

## Step 6: Install everything (once per machine, about 10–15 minutes)

```bash
bash cloud/setup_gpu.sh
```

The last lines should say **`environment OK`** and **`Setup done`**.

If it fails, copy the last 30 lines of the terminal and send them to me.

## Step 7: Start the run

Copy all three lines together:

```bash
mkdir -p results
POTHOLEMIX_ROOT=data/raw/pothole-mix NPD_ROOT=data/raw/NPD \
  nohup bash cloud/run_all.sh > results/run.log 2>&1 &
```

It now runs in the background. It keeps going even if you close the browser tab.

## Step 8: Watch progress (optional)

```bash
tail -f results/run.log
```

You'll see lines like `[37/186] ... eta 12.3 min`. Press **Ctrl + C** to stop *watching*. The run keeps going.

The run is finished when the log ends with **`ALL DONE`**.

## Step 9: Save the results to GitHub

```bash
git add results/*.csv results/tables results/gpu.txt
git commit -m "Benchmark results"
git push
```

Then tell me, and I'll read the results from GitHub and write up the findings.

If the push asks for a login and that doesn't work, use another way to get the files to me. In the Explorer, right-click the `results` folder, choose **Download**, and send it to me.

## Step 10: STOP THE MACHINE

On the GPU platform, **stop or terminate** the machine. You pay by the hour while it is on, even if it is doing nothing.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| A setup error that mentions CUDA or `torch` | Run `TORCH_CUDA=cu118 bash cloud/setup_gpu.sh` |
| `set POTHOLEMIX_ROOT` or `set NPD_ROOT` | You skipped Step 5, or the folders have other names. Run `ls data/raw` and send me the output. |
| The run stopped half-way | Run the Step 7 command again. It skips the work that's already done. |
| Anything else | Send me the last 30 lines of `results/run.log`. |

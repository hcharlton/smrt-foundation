import sys
import os
import subprocess

### config
DEV_BRANCH = "gefion-dev"
MAIN_BRANCH = "main"
BUNDLE = "gefion.bundle"

GEFION_OUTBOX = "/dcai/projects/cu_0030/xfer/chache/outbox"

RCLONE_REMOTE = "gefion"       
SFTP_FOLDER = "from_gefion"

def run(cmd):
    print(f"> {cmd}")
    if subprocess.call(cmd, shell=True) != 0:
        print("Command failed")
        sys.exit(1)

if len(sys.argv) != 2 or sys.argv[1] not in ["gefion", "genomedk"]:
    print("Usage: python sync.py [gefion | genomedk]")
    sys.exit(1)

mode = sys.argv[1]

### gefion: create bundle and move to outbox
if mode == "gefion":
    # sync with origin so the bundle is small
    run("git fetch origin")
    
    # create the bundle
    run(f"git bundle create {BUNDLE} {DEV_BRANCH} ^origin/main")
    
    # Move to outbox
    run(f"cp {BUNDLE} {GEFION_OUTBOX}/")
    
    # Cleanup local file
    if os.path.exists(BUNDLE):
        os.remove(BUNDLE)
    print("Export complete.")

### genomeDK mode: fetch the bundle and 
elif mode == "genomedk":
    print(f'Downloading bundle from {RCLONE_REMOTE}...')
    run(f"rclone copy {RCLONE_REMOTE}:{SFTP_FOLDER}/{BUNDLE} .")
    if not os.path.exists(BUNDLE):
        print(f"File '{BUNDLE}' not found (download failure?).")
        sys.exit(1)

    # update main
    run(f"git checkout {MAIN_BRANCH}")
    run(f"git pull origin {MAIN_BRANCH}")

    # verify/fetch bundle
    run(f"git bundle verify {BUNDLE}")
    run(f"git fetch {BUNDLE} {DEV_BRANCH}:refs/remotes/bundle/{DEV_BRANCH}")

    # force local dev branch to match bundle
    run(f"git checkout -B {DEV_BRANCH} refs/remotes/bundle/{DEV_BRANCH}")

    # merge into main and push
    run(f"git checkout {MAIN_BRANCH}")
    run(f"git merge --no-squash {DEV_BRANCH}")
    run(f"git push origin {MAIN_BRANCH}")

    # delete the branch, remove the bundle
    run(f"git branch -D {DEV_BRANCH}")
    if os.path.exists(BUNDLE):
        os.remove(BUNDLE)
    print("Import complete.")

### todo: update mode (for gefion the next day)
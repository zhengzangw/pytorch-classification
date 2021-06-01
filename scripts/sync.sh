set -e

# auto commit
# git add -A
# git commit --allow-empty -m "[auto] auto commit by script when pushed to server"

# push to server
rsync -ravh --include '.env' --exclude-from='.gitignore' . $@

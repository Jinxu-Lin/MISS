git rev-list --objects --all | sort -k 2 | \
while read hash filename; do
  echo "$(git cat-file -s $hash) $filename"
done | sort -n | tail -n 10
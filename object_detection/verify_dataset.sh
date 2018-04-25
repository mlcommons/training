if md5sum -c hashes.md5
then
  echo "PASSED"
else
  echo "FAILED"
fi

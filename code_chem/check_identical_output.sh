#!/bin/bash
set -e

f1="/fred/oz304/beckers/Chemistry/evolving_model/Crich/output/model_output_21549.dat"
f2="/fred/oz304/beckers/Chemistry/evolving_model_old/Crich/output/model_output_21549_old.dat"

ls -l "$f1" "$f2"
wc -l "$f1" "$f2"
sha256sum "$f1" "$f2"

if cmp -s "$f1" "$f2"; then
  echo "CMP_RESULT: IDENTICAL"
else
  echo "CMP_RESULT: DIFFERENT"
  echo "--- FIRST DIFF LINES (unified, first 120 lines) ---"
  diff -u "$f2" "$f1" | sed -n '1,120p'
fi
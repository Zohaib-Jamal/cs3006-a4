#!/usr/bin/env bash
# CS3006 Assignment 4 -- correctness sweep. Run from the repository root.
if [ "$(uname -m)" = "x86_64" ]; then SFX=_x86; else SFX=; fi
bad=0
ok() { "$@" > /dev/null 2>&1 && echo PASS || echo FAIL; }
row() { printf '%-44s mine=%s ref=%s\n' "$1" "$2" "$3"
        if [ "$2" = FAIL ] && [ "$3" = PASS ]; then bad=$((bad+1)); fi; }

cd scan
for t in scan find_repeats; do for i in ones random; do
  for n in 1 2 3 255 256 257 1000 65536 1000003; do
    row "$t -i $i -n $n" "$(ok ./cudaScan -m $t -i $i -n $n)" \
                         "$(ok ./cudaScan_ref$SFX -m $t -i $i -n $n)"
  done; done; done

cd ../render
for s in 256 512 768 1024; do
  for sc in rgb rgby rand10k rand100k rand1M micro2M biglittle littlebig \
            pattern snowsingle snow bouncingballs hypnosis fireworks; do
    for seed in 0 1 2; do
      row "$sc -s $s -S $seed" \
          "$(ok ./render -c -s $s -S $seed -f /tmp/sweep $sc)" \
          "$(ok ./render_ref$SFX -c -s $s -S $seed -f /tmp/sweep $sc)"
    done; done; done

echo "Cases where the reference passes and yours fails: $bad"

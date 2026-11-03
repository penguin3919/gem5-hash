#!/usr/bin/env bash
# Pack the matrix results:
#   results_matrix/<alloc>_<access>_<MiB>_<pt>_{stats.txt,simout.txt,config.ini,simerr.txt}
#   results_matrix/trace_<alloc>_<access>_<MiB>_<pt>.log
#   results_matrix/summary.csv   alloc,access,size_mib,structure,K,buckets,ov_buckets,status,simTicks
# then results_matrix.tgz.  Run from the gem5 root.
set -u
OUT=results_matrix
mkdir -p "$OUT"
echo "alloc,access,size_mib,structure,K,buckets,ov_buckets,status,simTicks" > $OUT/summary.csv
for d in m5out_mx_*; do
    [[ -d $d ]] || continue
    name=${d#m5out_mx_}                        # seq_rand_512_v6k4_24576
    IFS=_ read -r al ac n pt nb <<< "$name"
    case $pt in
        base|sv39) st=$pt; k=; nb=; ov= ;;
        v5)  st=v5; k=1 ;;
        v6k*) st=v6; k=${pt#v6k} ;;
    esac
    if [[ -n $nb ]]; then ov=$(awk -F= '/^hashPTOvBuckets=/ {print $2; exit}' $d/config.ini 2>/dev/null); fi
    if [[ -f $d/stats.txt ]] && grep -q '^simTicks' $d/stats.txt; then
        status=ok; ticks=$(awk '/^simTicks/ {print $2; exit}' $d/stats.txt)
    elif grep -q 'fatal:' $d/simerr 2>/dev/null; then status=fatal; ticks=
    else status=missing; ticks=; fi
    echo "$al,$ac,$n,$st,$k,$nb,$ov,$status,$ticks" >> $OUT/summary.csv
    for pair in stats.txt:stats.txt simout:simout.txt config.ini:config.ini simerr:simerr.txt; do
        src=${pair%%:*}; dst=${pair#*:}
        [[ -f $d/$src ]] && cp "$d/$src" "$OUT/${name}_${dst}"
    done
done
for d in m5out_mxtrace_*; do
    [[ -f $d/trace.log ]] && cp "$d/trace.log" "$OUT/trace_${d#m5out_mxtrace_}.log"
done
if command -v column >/dev/null; then column -s, -t $OUT/summary.csv; else cat $OUT/summary.csv; fi
tar czf results_matrix.tgz "$OUT"
echo "packed: results_matrix.tgz ($(du -h results_matrix.tgz | cut -f1))"

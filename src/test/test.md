============================================================
📊  BENCHMARK SUMMARY
============================================================

🏁  SPEED RANKING — ALL (tok/s)
  #    name                                     tok/s
  ---- ----------------------------------- ----------
  1    gen5_non_early_stop_paged               112.82
  2    gen1_non_early_stop_paged               108.98
  3    gen5_non_early_stop                     106.63
  4    gen7_paged                              106.13
  5    gen1_paged                              105.90
  6    gen1_non_early_stop                     105.09
  7    gen5_paged                              104.43
  8    gen3_paged                              104.16
  9    gen7                                    103.71
  10   gen5                                    103.53
  11   gen1                                     54.13
  12   gen3                                     53.30
  13   gen7_non_early_stop_paged                49.80
  14   gen7_non_early_stop                      49.01
  15   gen3_non_early_stop_paged                48.80
  16   gen3_non_early_stop                      47.10

📏  LENGTH RANKING — ALL (tokens)
  #    name                                    tokens
  ---- ----------------------------------- ----------
  1    gen1_non_early_stop                      55.12
  2    gen5_non_early_stop                      55.12
  3    gen1_non_early_stop_paged                55.12
  4    gen5_non_early_stop_paged                55.12
  5    gen1                                     28.00
  6    gen3                                     27.85
  7    gen5                                     24.69
  8    gen1_paged                               24.69
  9    gen5_paged                               24.69
  10   gen3_non_early_stop                      24.46
  11   gen7                                     24.46
  12   gen7_non_early_stop                      24.46
  13   gen3_paged                               24.46
  14   gen3_non_early_stop_paged                24.46
  15   gen7_paged                               24.46
  16   gen7_non_early_stop_paged                24.46

============================================================
📋  GROUP AVERAGES
============================================================

⚡  Avg Speed by group (tok/s)
  group                             tok/s  n
  ---------------------------- ----------  -
  paged                            105.15  (4)
  non_early_stop_paged              80.10  (4)
  normal                            78.67  (4)
  non_early_stop                    76.96  (4)

📐  Avg Length by group (tokens)
  group                            tokens  n
  ---------------------------- ----------  -
  non_early_stop                    39.79  (4)
  non_early_stop_paged              39.79  (4)
  normal                            26.25  (4)
  paged                             24.58  (4)

============================================================
🔬  SUB-DIMENSION COMPARISONS
============================================================

⚡  Speed: early_stop vs non_early_stop
  dimension                           tok/s  n
  ------------------------------ ----------  -
  early_stop                          91.91  8
  non_early_stop                      78.53  8

📐  Length: early_stop vs non_early_stop
  dimension                          tokens  n
  ------------------------------ ----------  -
  non_early_stop                      39.79  8
  early_stop                          25.41  8

⚡  Speed: paged vs non_paged
  dimension                           tok/s  n
  ------------------------------ ----------  -
  paged                               92.63  8
  non_paged                           77.81  8

📐  Length: paged vs non_paged
  dimension                          tokens  n
  ------------------------------ ----------  -
  non_paged                           33.02  8
  paged                               32.18  8

⚡  Speed: no_cache vs kv_cache
  dimension                           tok/s  n
  ------------------------------ ----------  -
  kv_cache gen5-8                     92.01  8
  no_cache gen1-4                     78.43  8

📐  Length: no_cache vs kv_cache
  dimension                          tokens  n
  ------------------------------ ----------  -
  no_cache gen1-4                     33.02  8
  kv_cache gen5-8                     32.18  8

⚡  Speed: score_fn1 vs score_fn2
  dimension                           tok/s  n
  ------------------------------ ----------  -
  score_fn1 gen1,2,5,6               100.19  8
  score_fn2 gen3,4,7,8                70.25  8

📐  Length: score_fn1 vs score_fn2
  dimension                          tokens  n
  ------------------------------ ----------  -
  score_fn1 gen1,2,5,6                40.32  8
  score_fn2 gen3,4,7,8                24.88  8

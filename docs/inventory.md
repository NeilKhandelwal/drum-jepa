# E-GMD MIDI inventory (drum map v1)

- Files: 45537 total, 4552 unique by content, 0 unreadable
- Unique duration: 81.1 h; note-ons: 2,652,230

## Pitches

| pitch | count | share | V1 class | GMD description |
|---:|---:|---:|---|---|
| 38 | 641,961 | 24.20% | snare | Snare (Head) |
| 36 | 537,359 | 20.26% | kick | Kick |
| 44 | 258,785 | 9.76% | hh_pedal | HH Pedal |
| 51 | 255,694 | 9.64% | ride | Ride (Bow) |
| 40 | 151,432 | 5.71% | rimshot | Snare (Rim) |
| 22 | 146,153 | 5.51% | hh_closed | HH Closed (Edge) |
| 54 | 141,186 | 5.32% | **UNMAPPED** |  |
| 42 | 132,065 | 4.98% | hh_closed | HH Closed (Bow) |
| 43 | 69,607 | 2.62% | tom_lo | Tom 3 |
| 48 | 66,933 | 2.52% | tom_hi | Tom 1 |
| 37 | 57,767 | 2.18% | crossstick | Snare X-Stick |
| 53 | 48,604 | 1.83% | ride_bell | Ride (Bell) |
| 26 | 38,331 | 1.45% | hh_open | HH Open (Edge) |
| 45 | 26,465 | 1.00% | tom_mid | Tom 2 |
| 55 | 17,557 | 0.66% | crash1 | Crash 1 (Edge) |
| 59 | 12,712 | 0.48% | ride | Ride (Edge) |
| 50 | 10,644 | 0.40% | tom_hi | Tom 1 (Rim) |
| 46 | 8,995 | 0.34% | hh_open | HH Open (Bow) |
| 39 | 7,768 | 0.29% | **UNMAPPED** |  |
| 58 | 6,323 | 0.24% | tom_lo | Tom 3 (Rim) |
| 52 | 5,687 | 0.21% | crash2 | Crash 2 (Edge) |
| 47 | 3,462 | 0.13% | tom_mid | Tom 2 (Rim) |
| 49 | 3,178 | 0.12% | crash1 | Crash 1 (Bow) |
| 56 | 2,155 | 0.08% | **UNMAPPED** |  |
| 57 | 1,407 | 0.05% | crash2 | Crash 2 (Bow) |

**3 unmapped pitches** — add to drum_map.py or justify dropping.

**Decision (2026-09-04):** pitches 54, 39 and 56 are not new drum zones. They
are per-kit pad remaps in the MIDI that the TD-17 emitted while re-recording
each kit (54 = hi-hat zones on tambourine-hat kits, 56 = tom-2 rim on cowbell
kits, 39 = rim zones on kits without rim sounds); see the per-kit deviation
table at the end of this file. Hit counts are conserved under every remap, so
the action is built from the **canonical** MIDI of each sequence (identical
across 25 kits, e.g. `Acoustic Kit`), never from the kit's own file. The
canonical MIDI contains no unmapped pitch, so DRUM_MAP_V1 stays at v1.
Consequence: the "same action, different kit" counterfactual holds at the
pad level for all 43 kits, but on remapping kits the kit's pad->sound map is
many-to-one (extreme case: all five hi-hat pitches -> one sound). That is a
kit property the state must carry, not an action difference.


## Classes (V1)

| class | count | share |
|---|---:|---:|
| kick | 537,359 | 20.26% |
| snare | 641,961 | 24.20% |
| rimshot | 151,432 | 5.71% |
| crossstick | 57,767 | 2.18% |
| tom_hi | 77,577 | 2.92% |
| tom_mid | 29,927 | 1.13% |
| tom_lo | 75,930 | 2.86% |
| hh_closed | 278,218 | 10.49% |
| hh_open | 47,326 | 1.78% |
| hh_pedal | 258,785 | 9.76% |
| crash1 | 20,735 | 0.78% |
| crash2 | 7,094 | 0.27% |
| ride | 268,406 | 10.12% |
| ride_bell | 48,604 | 1.83% |

## Chokes (aftertouch)

No polyphonic aftertouch found. Chokes are not encoded; the choke eval is off the table.

Channel aftertouch messages: 0

## Hi-hat pedal position (CC 4)

- events: 3,115,353
- value 0-15: 15.3%  |  16-111: 84.7%  |  112-127: 0.0%
- verdict: Substantial mid-range mass — position may carry continuous information; inspect the histogram.

Other CC numbers seen: 4: 3,115,353

## Onset density per 2 s window

| onsets in window | windows | share |
|---|---:|---:|
| 0-0 | 7,411 | 5.0% |
| 1-2 | 977 | 0.7% |
| 3-5 | 1,584 | 1.1% |
| 6-10 | 10,389 | 7.0% |
| 11-20 | 71,741 | 48.4% |
| 21-40 | 56,120 | 37.8% |
| 41-∞ | 153 | 0.1% |

## Velocity by class (16 bins of 8)

| class | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 | 72 | 80 | 88 | 96 | 104 | 112 | 120 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kick | 4,411 | 75,733 | 23,266 | 47,889 | 70,698 | 83,708 | 54,239 | 38,667 | 31,494 | 22,380 | 13,260 | 9,086 | 7,543 | 6,819 | 5,762 | 42,404 |
| snare | 18,434 | 53,315 | 47,932 | 65,124 | 86,177 | 84,116 | 55,179 | 33,619 | 27,265 | 21,849 | 18,517 | 17,026 | 16,560 | 13,611 | 12,725 | 70,512 |
| rimshot | 20 | 446 | 896 | 1,144 | 2,135 | 2,055 | 1,965 | 1,875 | 2,563 | 2,846 | 3,696 | 10,251 | 16,413 | 15,222 | 14,810 | 75,095 |
| crossstick | 50 | 1,080 | 2,291 | 2,958 | 5,821 | 5,514 | 5,059 | 4,738 | 6,551 | 7,503 | 10,040 | 4,546 | 315 | 303 | 262 | 736 |
| tom_hi | 31 | 193 | 261 | 587 | 1,283 | 2,819 | 4,869 | 5,426 | 6,381 | 6,449 | 6,370 | 7,430 | 7,486 | 5,615 | 4,679 | 17,698 |
| tom_mid | 17 | 179 | 119 | 134 | 397 | 771 | 1,394 | 1,710 | 2,210 | 2,874 | 3,507 | 3,022 | 3,112 | 2,264 | 2,123 | 6,094 |
| tom_lo | 0 | 131 | 317 | 638 | 1,587 | 2,705 | 3,744 | 4,374 | 6,112 | 6,710 | 7,470 | 7,805 | 6,886 | 5,305 | 4,553 | 17,593 |
| hh_closed | 1,309 | 6,064 | 12,812 | 21,853 | 27,856 | 31,118 | 26,062 | 22,325 | 22,104 | 18,697 | 14,448 | 12,949 | 10,696 | 8,649 | 7,375 | 33,901 |
| hh_open | 79 | 676 | 523 | 1,093 | 1,825 | 2,024 | 2,562 | 3,067 | 3,306 | 3,044 | 2,634 | 2,499 | 2,448 | 2,410 | 1,854 | 17,282 |
| hh_pedal | 0 | 0 | 13,417 | 13,774 | 12,344 | 25,768 | 25,819 | 32,269 | 35,669 | 30,844 | 31,196 | 20,881 | 10,241 | 3,338 | 1,139 | 2,086 |
| crash1 | 352 | 991 | 219 | 162 | 360 | 537 | 561 | 797 | 1,190 | 1,530 | 2,134 | 2,687 | 2,583 | 2,105 | 1,065 | 3,462 |
| crash2 | 12 | 10 | 32 | 74 | 205 | 270 | 330 | 255 | 344 | 564 | 610 | 591 | 600 | 528 | 532 | 2,137 |
| ride | 725 | 1,733 | 3,654 | 9,406 | 23,469 | 34,292 | 40,039 | 32,611 | 27,637 | 18,969 | 15,443 | 11,746 | 10,586 | 8,024 | 6,345 | 23,727 |
| ride_bell | 0 | 0 | 135 | 678 | 1,543 | 2,447 | 2,652 | 3,683 | 3,867 | 3,222 | 2,779 | 3,036 | 2,429 | 2,465 | 2,270 | 17,398 |

## Per-kit MIDI deviation from the canonical performance

Canonical = pitch histogram shared by the most kits for that sequence.
`kept` = share of canonical note-ons present at the same pitch on this kit.

| kit | canonical seqs | kept | extra pitches (count) | missing (share of that pitch) |
|---|---:|---:|---|---|
| Acoustic Kit | 1059/1059 | 100.0% | - | - |
| Studio (Live Room) | 1059/1059 | 100.0% | - | - |
| Classic Rock | 1059/1059 | 100.0% | - | - |
| ClassicMetal (80s-90s) | 1059/1059 | 100.0% | - | - |
| 60s Rock | 1059/1059 | 100.0% | - | - |
| Modern Funk | 1059/1059 | 100.0% | - | - |
| Raw Dnb (Layered Hybrid) | 1059/1059 | 100.0% | - | - |
| Fat Rock (Power Toms) | 1059/1059 | 100.0% | - | - |
| Pop-Rock (Studio) | 1059/1059 | 100.0% | - | - |
| Dry & Heavy (Folk Rock) | 1059/1059 | 100.0% | - | - |
| Heavy Metal | 1059/1059 | 100.0% | - | - |
| Arena Stage | 1059/1059 | 100.0% | - | - |
| Jazz | 1059/1059 | 100.0% | - | - |
| More Cowbell (Pop-Rock) | 1059/1059 | 100.0% | - | - |
| Live Rock | 1059/1059 | 100.0% | - | - |
| Shuffle (Blues) | 1059/1059 | 100.0% | - | - |
| Alternative (METAL) | 1059/1059 | 100.0% | - | - |
| Rockin Gate (80s) | 1059/1059 | 100.0% | - | - |
| Live Fusion | 1059/1059 | 100.0% | - | - |
| Speed Metal | 1059/1059 | 100.0% | - | - |
| Cassette (Lo-Fi Compress) | 1059/1059 | 100.0% | - | - |
| Bigga Bop (Jazz) | 1059/1059 | 100.0% | - | - |
| Alternative (Rock) | 1059/1059 | 100.0% | - | - |
| Tight Prog | 1059/1059 | 100.0% | - | - |
| Unplugged | 1059/1059 | 100.0% | - | - |
| JingleStacks (2nd Hi-Hat) | 905/1059 | 99.6% | 42 (1,222) | 47 Tom 2 (Rim): 100% |
| Jazz Funk | 905/1059 | 99.6% | 56 (1,222) | 47 Tom 2 (Rim): 100% |
| Second Line | 905/1059 | 99.6% | 56 (1,222) | 47 Tom 2 (Rim): 100% |
| Warmer Funk | 905/1059 | 99.6% | 56 (1,222) | 47 Tom 2 (Rim): 100% |
| Super Boom (Layered) | 905/1059 | 99.6% | 39 (1,222) | 47 Tom 2 (Rim): 100% |
| Funk Rock | 905/1059 | 99.6% | 56 (1,222) | 47 Tom 2 (Rim): 100% |
| West Coast (FUNK) | 822/1059 | 99.4% | 42 (1,222), 56 (933) | 47 Tom 2 (Rim): 100%, 58 Tom 3 (Rim): 100% |
| 808 Simple | 729/1059 | 97.3% | 40 (9,011) | 37 Snare X-Stick: 100% |
| 909 Simple | 729/1059 | 97.3% | 40 (9,011) | 37 Snare X-Stick: 100% |
| Custom1 | 729/1059 | 97.3% | 40 (9,011) | 37 Snare X-Stick: 100% |
| Custom2 | 729/1059 | 97.3% | 40 (9,011) | 37 Snare X-Stick: 100% |
| Custom3 | 729/1059 | 97.3% | 40 (9,011) | 37 Snare X-Stick: 100% |
| Deep Daft | 603/1059 | 96.9% | 39 (1,222), 54 (9,085) | 26 HH Open (Edge): 100%, 46 HH Open (Bow): 100%, 47 Tom 2 (Rim): 100% |
| Nu RNB | 562/1059 | 96.5% | 37 (495), 38 (1,222), 39 (933), 40 (9,011) | 37 Snare X-Stick: 89%, 47 Tom 2 (Rim): 100%, 50 Tom 1 (Rim): 100%, 58 Tom 3 (Rim): 100% |
| Ele-Drum | 562/1059 | 96.5% | 37 (495), 39 (2,155), 40 (9,011) | 37 Snare X-Stick: 89%, 47 Tom 2 (Rim): 100%, 50 Tom 1 (Rim): 100%, 58 Tom 3 (Rim): 100% |
| Dark Hybrid | 293/1059 | 71.3% | 54 (95,622) | 22 HH Closed (Edge): 100%, 26 HH Open (Edge): 100%, 42 HH Closed (Bow): 100%, 44 HH Pedal: 100%, 46 HH Open (Bow): 100% |
| Big Room (Layered) | 290/1059 | 93.7% | 36 (933), 38 (17,421), 39 (2,683) | 40 Snare (Rim): 100%, 47 Tom 2 (Rim): 100%, 50 Tom 1 (Rim): 100%, 58 Tom 3 (Rim): 100% |
| Compact Lite (w/ Tambourine HH) | 260/1059 | 71.0% | 54 (96,844) | 22 HH Closed (Edge): 100%, 26 HH Open (Edge): 100%, 42 HH Closed (Bow): 100%, 44 HH Pedal: 100%, 46 HH Open (Bow): 100%, 47 Tom 2 (Rim): 100% |

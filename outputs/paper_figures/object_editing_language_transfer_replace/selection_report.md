# Actual language-selected editing

External controller: text-only Qwen3-VL-4B-Instruct. No image/GT/reference IDs or manual correction.
All actual decisions below, including abstentions; not a benchmark accuracy estimate.

| Request | Command | Actual operation | Selected A / B | Selected categories A / B | Generation |
| --- | --- | --- | --- | --- | --- |
| edit_001 | Remove the cup from image A. | remove | obj_a32d6bf62dbd / — | Cup / — | generated |
| edit_002 | Transfer the brown dog's appearance from image B to the dog in image A. | transfer | obj_e742cde13bfd / obj_f2c2ca473dd6 | Dog / Dog | generated |
| edit_003 | Remove the bench from image A. | remove | obj_1a47f44f2598 / — | Bench / — | generated |
| edit_004 | Replace the snowboard in image A with the surfboard from image B. | replace | obj_af492ec5af8f / obj_c6456e937620 | Snowboard / Surfboard | generated |
| edit_005 | Remove the giraffe from image A. | remove | obj_28d369b1e0c0 / — | Giraffe / — | generated |
| edit_006 | Transfer the refrigerator's appearance from image B to the refrigerator in image A. | transfer | obj_1eb4b22d65ba / obj_866188538541 | Refrigerator / Refrigerator | generated |
| edit_007 | Remove the airplane from image A. | remove | obj_2bff221c0d13 / — | Pizza / — | generated |
| edit_008 | Remove the dog from image A. | remove | obj_c87f2b158516 / — | Dog / — | generated |

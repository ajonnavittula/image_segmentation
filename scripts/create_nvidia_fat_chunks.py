import json
import os

chunks = {}

chunks["1"] = ['nvidia_fat_002_master_chef_can_16k_train', 'nvidia_fat_003_cracker_box_16k_train', 'nvidia_fat_004_sugar_box_16k_train', 
'nvidia_fat_005_tomato_soup_can_16k_train', 'nvidia_fat_006_mustard_bottle_16k_train', 'nvidia_fat_007_tuna_fish_can_16k_train', 
'nvidia_fat_008_pudding_box_16k_train', 'nvidia_fat_009_gelatin_box_16k_train', 'nvidia_fat_010_potted_meat_can_16k_train', 
'nvidia_fat_011_banana_16k_train', 'nvidia_fat_019_pitcher_base_16k_train', 'nvidia_fat_021_bleach_cleanser_16k_train', 
'nvidia_fat_024_bowl_16k_train', 'nvidia_fat_025_mug_16k_train', 'nvidia_fat_035_power_drill_16k_train', 'nvidia_fat_036_wood_block_16k_train', 
'nvidia_fat_037_scissors_16k_train', 'nvidia_fat_040_large_marker_16k_train', 'nvidia_fat_051_large_clamp_16k_train', 
'nvidia_fat_052_extra_large_clamp_16k_train', 'nvidia_fat_061_foam_brick_16k_train', 'nvidia_fat_kitchen_0_train', 'nvidia_fat_kitchen_1_train', 
'nvidia_fat_kitchen_2_train', 'nvidia_fat_kitchen_3_train', 'nvidia_fat_kitchen_4_train', 'nvidia_fat_kitedemo_0_train', 'nvidia_fat_kitedemo_1_train', 
'nvidia_fat_kitedemo_2_train', 'nvidia_fat_kitedemo_3_train', 'nvidia_fat_kitedemo_4_train', 'nvidia_fat_temple_0_train', 'nvidia_fat_temple_1_train', 
'nvidia_fat_temple_2_train', 'nvidia_fat_temple_3_train', 'nvidia_fat_temple_4_train']

with open("nvidia_fat_chunks.json", "w") as jsonfile:
    json.dump(chunks, jsonfile, sort_keys=True, indent=4)
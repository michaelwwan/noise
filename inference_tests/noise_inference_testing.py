# Utilize Pytest

# Run from CLI from /home/rneilson/VSCode/Noise/Inference_Testing_Env using pytest noise/tests/noise_inference_testing.py

import pytest

import noise.noise_inference as inference_functions # This will import all functions within the noise_inference script 

# Below are unit test that can be performed in the noise_inference script 


def test_masking_coordinates_to_list():

    pass 

def test_calculate_pixel_area():

    pass 

def test_total_area_per_well():

    pass

def test_percent_ocl_area_per_well():

    # Test 50/100 = 50% Area
    assert inference_functions.percent_ocl_area_per_well(50, 100) == 50

    # Test 0/50 = 0%
    assert inference_functions.percent_ocl_area_per_well(0, 50) == 0

# Need to figure out how to test the actual inference part of the script, end to end testing? 









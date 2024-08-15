#!/bin/bash

# Run the command with cohort 'hc'
python daily_living_results_eval.py --cohort hc --model segmentation

# Run the command with cohort 'hd'
python daily_living_results_eval.py --cohort hd --model segmentation

# Run the command with cohort 'hc'
python daily_living_results_eval.py --cohort hc --model classification

# Run the command with cohort 'hd'
python daily_living_results_eval.py --cohort hd --model classification

"""The model loops are set up so that the last element is not processed, as there would be no
further element available to compare against and compute loss. To ensure all data is processed
during TBPTT, segments `x` fed into successive computations of the graph should overlap by 1.
Give this constant a name to make it less magical.
"""
LEFTOVER = 1

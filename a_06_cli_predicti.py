#!/usr/bin/env python

"""Building a click command line interface that 
takes in a input text and return the predicts whether it shows suicide attempy """

import click
from a_05_classification import classify_newtext


# Create a click command that takes in a line of text as input and returns the predicted classification of suicide attempy
@click.command()
@click.option("--text", prompt="enter or paste your text", help="text for analysis")
def main(text):
    classify_newtext(text)


# Run the cli
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()

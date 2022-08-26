#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
from operator import concat
import wandb
import numpy as np
import pandas as pd
#from ...components.wandb_utils.log_artifact import log_artifact


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    # download the training data as csv file
    logger.debug(f"Downloading data set {args.input_artifact} as file")
    local_path = wandb.use_artifact(args.input_artifact).file()
    # read it and provide it as pandas dataframe
    logger.debug(f"Reading data set file {args.input_artifact}")
    df = pd.read_csv(local_path)

    # Drop outliers of price feature
    logger.debug("Drop outliers of price feature")
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.debug("Convert last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Transform object features to categorical features
    logger.debug("Transform object features to categorical features")
    df = pd.concat([
        df.select_dtypes([], ['object']),
        df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
        ], axis=1)#.reindex_axis(df.columns, axis=1)

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info(f"Returning cleaned data of {args.input_artifact}")
    logger.info(f"Uploading {args.output_artifact} to Weights & Biases")
    file_name_clean_sample = "clean_sample.csv"
    df.to_csv(file_name_clean_sample,index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(file_name_clean_sample)
    run.log_artifact(artifact)
    # We need to call this .wait() method before we can use the
    # version below. This will wait until the artifact is loaded into W&B and a
    # version is assigned
    artifact.wait()

"""     log_artifact(
        args.output_artifact,
        args.output_type,
        args.output_description,
        "clean_sample.csv",
        run,
    ) """


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str, ## INSERT TYPE HERE: str, float or int,
        help="Name for the input artifact", ## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str, ## INSERT TYPE HERE: str, float or int,
        help="Name for the output artifact", ## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str, ## INSERT TYPE HERE: str, float or int,
        help="Type of the output artifact. This will be visible in W&B.", ## INSERT DESCRIPTION HERE,
        default="",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str, ## INSERT TYPE HERE: str, float or int,
        help="Description of the output artifact. This will be visible in W&B.", ## INSERT DESCRIPTION HERE,
        default="",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float, ## INSERT TYPE HERE: str, float or int,
        help="Minimum allowed price. Lower prices in the data set will be set to min_price.", ## INSERT DESCRIPTION HERE,
        default=10,
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float, ## INSERT TYPE HERE: str, float or int,
        help="Maximum allowed price. Higher prices in the data set will be set to max_price.", ## INSERT DESCRIPTION HERE,
        default=350,
        required=True
    )


    args = parser.parse_args()

    go(args)

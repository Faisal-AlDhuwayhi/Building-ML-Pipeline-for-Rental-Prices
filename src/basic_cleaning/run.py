#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info(f'Download input artifact from W&B: {args.input_artifact}')
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df_sample = pd.read_csv(artifact_local_path)
    
    # remove price variable outliers
    logger.info(f'Remove price (variable) outliers by limiting to these thresholds: min={args.min_price}, max={args.max_price}')
    min_price, max_price  = args.min_price, args.max_price
    idx = df_sample['price'].between(min_price, max_price)
    df_clean = df_sample[idx].copy()
    
    # convert 'last_review' variable from string to datetime
    logger.info('Convert variable "last_review" from string to datetime type')
    df_clean['last_review'] = pd.to_datetime(df_clean['last_review'])

    # Only keep rows that are in the proper geolocation of NYC
    logger.info('Only keep rows that are in the proper geolocation of NYC')
    idx = df_clean['longitude'].between(-74.25, -73.50) & df_clean['latitude'].between(40.5, 41.2)
    df_clean = df_clean[idx].copy()
    
    # store cleaned dataframe
    logger.info(f'Store cleaned dataframe in: {args.output_artifact}')
    df_clean.to_csv(args.output_artifact, index=False)

    # log artifact to Weights & Biases
    logger.info(f'Logging artifact to W&B: {args.output_artifact}') 
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact name (W&B) (e.g. sample.csv)",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output artifact name (W&B) (e.g. clean_sample.csv)",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Output artifact type (W&B)",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Output artifact description (W&B)",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price (variable) to filter the input artifact for cleaning (remove outliers) (e.g. 10)",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price (variable) to filter the input artifact for cleaning (remove outliers) (e.g. 350)",
        required=True
    )


    args = parser.parse_args()

    go(args)

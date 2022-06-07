import argparse
import logging
import os
import pickle
import traceback
from copy import deepcopy

import wandb
import yaml

from src.utils.cancer_simulation import get_cancer_sim_data
from src.utils.data_utils import process_data, read_from_file, write_to_file
from src.utils.process_irregular_data import *
from trainer import trainer

os.environ["WANDB_API_KEY"] = "ADD YOUR WANDB API KEY HERE"
wandb_entity = "ADD YOUR WANDB ENTITY HERE"


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemo_coeff", default=2, type=int)
    parser.add_argument("--radio_coeff", default=2, type=int)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--model_name", default="te_cde_test")
    parser.add_argument("--load_dataset", default=True)
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--use_transformed", default=True)
    parser.add_argument("--multistep", default=False)
    parser.add_argument("--kappa", type=int, default=10)
    parser.add_argument("--lambda_val", type=float, default=1)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--max_horizon", type=int, default=5)
    parser.add_argument("--save_raw_datapath", type=str, default=None)
    parser.add_argument("--save_transformed_datapath", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()
    if not os.path.exists("./tmp_models/"):
        os.mkdir("./tmp_models/")

    use_transformed = str(args.use_transformed) == "True"
    multistep = str(args.multistep) == "True"

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    strategy = "all"

    logging.info("WANDB init...")
    # start a new run
    run = wandb.init(
        project="te_cde_run",
        entity=wandb_entity,
        config=f"./experiments/{args.experiment}.yml",
    )

    config = wandb.config

    if args.data_path == None:
        logging.info("Generating dataset")
        pickle_map = get_cancer_sim_data(
            chemo_coeff=args.chemo_coeff,
            radio_coeff=args.radio_coeff,
            b_load=True,
            b_save=False,
            model_root=args.results_dir,
        )
    else:
        logging.info(f"Loading dataset from: {args.data_path}")
        pickle_map = read_from_file(args.data_path)

    wandb.log({"chemo_coeff": args.chemo_coeff})
    wandb.log({"radio_coeff": args.radio_coeff})

    kappa = int(args.kappa)
    wandb.log({"kappa": kappa})

    lambda_val = float(args.lambda_val)
    wandb.log({"lambda": lambda_val})

    max_samples = int(args.max_samples)
    wandb.log({"max_samples": max_samples})

    max_horizon = int(args.max_horizon)
    wandb.log({"max_horizon": max_horizon})

    wandb.log({"strategy": strategy})

    coeff = int(args.radio_coeff)

    if args.save_raw_datapath != None:
        logging.info(f"Writing raw data to {args.save_raw_datapath}")
        write_to_file(
            pickle_map,
            f"{args.save_raw_datapath}/new_cancer_sim_{coeff}_{coeff}.p",
        )

    if bool(use_transformed) == False:
        logging.info("Transforming dataset")
        pickle_map = transform_data(
            data=pickle_map,
            interpolate=False,
            strategy=strategy,
            sample_prop=config["sample_proportion"],
            kappa=kappa,
            max_samples=max_samples,
        )

    else:
        transformed_datapath = f"/content/drive/MyDrive/kappa{kappa}/new_cancer_sim_{coeff}_{coeff}_kappa_{kappa}.p"
        logging.info(f"Loading transformed data from {transformed_datapath}")
        pickle_map = read_from_file(transformed_datapath)

    if args.save_transformed_datapath != None:
        logging.info(f"Writing transformed data to {args.save_transformed_datapath}")
        write_to_file(
            pickle_map,
            f"{args.save_transformed_datapath}/new_cancer_sim_{coeff}_{coeff}_kappa_{kappa}.p",
        )

    logging.info("Processing dataset")
    training_processed, validation_processed, test_processed = process_data(pickle_map)

    use_time = config["use_time"]

    done = False
    tries = 0
    while done == False:
        if tries > 10:
            done = True
            break

        try:
            logging.info("Training model...")
            cde_trainer = trainer(
                run=run,
                hidden_channels_x=config["hidden_channels_x"],
                hidden_channels_a=config["hidden_channels_a"],
                output_channels=config["output_channels"],
                sample_proportion=config["sample_proportion"],
                use_time=config["use_time"],
                lambda_val=lambda_val,
            )

            wandb.log({"proportion": config["sample_proportion"]})
            cde_trainer.fit(
                train_data=training_processed,
                validation_data=validation_processed,
                epochs=config["epochs"],
                patience=config["patience"],
                batch_size=config["batch_size"],
            )
            logging.info("Testing model...")
            cde_trainer.predict(test_data=test_processed)
            done = True

        except Exception as e:
            print(e)
            tries = tries + 1

    if bool(multistep) == True:
        multidone = False
        multitries = 0

        while multidone == False:
            if multitries > 10:
                multidone = True
                break
            try:
                logging.info("Fitting multistep model...")
                cde_trainer.fit_multistep(
                    train_data=training_processed,
                    validation_data=validation_processed,
                    epochs=config["epochs"],
                    patience=config["patience"],
                    batch_size=config["batch_size"],
                    max_horizon=max_horizon,
                )

                logging.info("Testing multistep model...")
                cde_trainer.multistep_predict(
                    test_data=test_processed,
                    max_horizon=max_horizon,
                )
                multidone = True

            except Exception as e:
                # to traceback where actual error takes place after multiple tries
                print(traceback.format_exc())
                multitries = multitries + 1

    run.finish()

    # remove tmp model from file system
    os.system("rm -rf ./tmp_models/")

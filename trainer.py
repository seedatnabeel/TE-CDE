import argparse
import logging
import os
import pickle
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchcde
import torchmetrics
import wandb
from torch.nn import functional as F
from tqdm import tqdm

from src.models.CDE_model import NeuralCDE
from src.utils.data_utils import (
    data_to_torch_tensor,
    data_to_torch_tensor_multistep,
    process_counterfactual_seq_test_data,
    process_seq_data,
    read_from_file,
    write_to_file,
)
from src.utils.losses import compute_norm_mse_loss
from src.utils.training_tools import EarlyStopping, enable_dropout


class trainer:
    def __init__(
        self,
        run,
        hidden_channels_x,
        hidden_channels_a,
        output_channels,
        sample_proportion=1,
        use_time=True,
        lambda_val=None,
    ):

        self.run = run
        self.model = None
        self.multistep_model = None

        self.device = None

        self.use_time = use_time
        self.hidden_channels_x = hidden_channels_x
        self.hidden_channels_a = hidden_channels_a
        self.output_channels = output_channels

        self.sample_proportion = sample_proportion
        self.lambda_val = lambda_val

        if self.use_time == False:
            self.time_concat = 0

    def _train(self, train_dataloader, model, optimizer, lambda_val):
        model = model.train()

        treatment_loss = nn.CrossEntropyLoss()  # nn.NLLLoss()

        train_losses_total = []
        train_losses_y = []
        train_losses_a = []

        for (
            batch_coeffs_x,
            batch_y,
            batch_treat,
        ) in train_dataloader:

            batch_coeffs_x = torch.tensor(
                batch_coeffs_x,
                dtype=torch.float,
                device=self.device,
            )

            outcomes = torch.tensor(
                batch_y[:, :, 0],
                dtype=torch.float,
                device=self.device,
            )
            active_entries = torch.tensor(
                batch_y[:, :, 1],
                dtype=torch.float,
                device=self.device,
            )
            current_treatment = torch.tensor(
                batch_treat[:, -1, :],
                dtype=torch.float,
                device=self.device,
            )

            # TODO ADD BACK .squeeze(-1)
            pred_y, pred_a_soft, pred_a, _ = model(batch_coeffs_x, self.device)

            # compute norm mse loss - outcomes loss
            loss_y = compute_norm_mse_loss(outcomes, pred_y, active_entries)

            # compute interventions/actions loss
            # loss_a = treatment_loss(pred_a_soft, torch.max(current_treatment, 1)[1])
            loss_a = treatment_loss(pred_a, torch.argmax(current_treatment, dim=1))

            # total_loss
            total_loss = loss_y + lambda_val * loss_a

            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_losses_total.append(total_loss.item())
            train_losses_y.append(loss_y.item())
            train_losses_a.append(loss_a.item())

        return model, train_losses_total, train_losses_y, train_losses_a

    def _test(self, test_dataloader, model, lambda_val):
        # eval mode
        model = model.eval()

        treatment_loss = nn.CrossEntropyLoss()  # nn.NLLLoss()

        test_losses_total = []
        test_losses_y = []
        test_losses_a = []

        with torch.no_grad():
            for (
                batch_coeffs_x_val,
                batch_y_val,
                batch_val_treat,
            ) in test_dataloader:

                batch_coeffs_x_val = torch.tensor(
                    batch_coeffs_x_val,
                    dtype=torch.float,
                    device=self.device,
                )

                outcomes_val = torch.tensor(
                    batch_y_val[:, :, 0],
                    dtype=torch.float,
                    device=self.device,
                )
                active_entries_val = torch.tensor(
                    batch_y_val[:, :, 1],
                    dtype=torch.float,
                    device=self.device,
                )
                current_treatment_val = torch.tensor(
                    batch_val_treat[:, -1, :],
                    dtype=torch.float,
                    device=self.device,
                )

                # TODO ADD BACK .squeeze(-1)
                pred_y_val, pred_a_val, pred_a, _ = model(
                    batch_coeffs_x_val,
                    self.device,
                )

                # compute norm mse loss - outcomes loss
                loss_y_val = compute_norm_mse_loss(
                    outcomes_val,
                    pred_y_val,
                    active_entries_val,
                )

                # compute interventions/actions loss
                # loss_a_val = treatment_loss(
                #     pred_a_val, torch.max(current_treatment_val, 1)[1]
                # )

                loss_a_val = treatment_loss(
                    pred_a,
                    torch.argmax(current_treatment_val, dim=1),
                )

                # total_loss
                total_loss_val = loss_y_val + lambda_val * loss_a_val

                test_losses_total.append(total_loss_val.item())
                test_losses_y.append(loss_y_val.item())
                test_losses_a.append(loss_a_val.item())

        return model, test_losses_total, test_losses_y, test_losses_a

    def prepare_dataloader(self, data, batch_size):
        data_X, data_A, data_Time, data_y, data_tr, _, _ = data_to_torch_tensor(
            data,
            sample_prop=self.sample_proportion,
        )

        data_concat = torch.cat((data_X, data_A), 2)

        data_shape = list(data_concat.shape)

        # to make it continuous we interpolate (linear - keep causal interpretation)
        coeffs = torchcde.linear_interpolation_coeffs(data_concat)

        # create train data_loader
        dataset = torch.utils.data.TensorDataset(coeffs, data_y, data_tr)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512)

        return dataloader, data_shape

    def prepare_dataloader_multistep(self, data, batch_size, max_horizon):
        data_X, data_A, data_Time, data_y, data_tr, _, _ = data_to_torch_tensor(
            data,
            sample_prop=self.sample_proportion,
        )

        if torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"

        device = torch.device(device_type)

        encode_concat = torch.cat((data_X, data_A), 2)

        # to make it continuous we interpolate (linear - keep causal interpretation)
        encode_coeffs = torchcde.linear_interpolation_coeffs(encode_concat)

        encode_coeffs = torch.tensor(encode_coeffs, dtype=torch.float, device=device)

        pred_y_test_sample, pred_a_test_sample, pred_a_test, z_hat_sample = self.model(
            encode_coeffs,
            device,
        )

        processed_data_multi = process_seq_data(
            data_map=data,
            states=z_hat_sample.cpu().detach().numpy(),
            projection_horizon=max_horizon,
        )

        data_X, data_A, data_Time, data_y, data_tr = data_to_torch_tensor_multistep(
            processed_data_multi,
            sample_prop=self.sample_proportion,
            max_horizon=max_horizon,
        )

        data_concat = torch.cat((data_X, data_A), 2)

        data_shape = list(data_concat.shape)

        # to make it continuous we interpolate (linear - keep causal interpretation)
        coeffs = torchcde.linear_interpolation_coeffs(data_concat)

        # create train data_loader
        dataset = torch.utils.data.TensorDataset(coeffs, data_y, data_tr)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512)
        return dataloader, data_shape

    def fit(self, train_data, validation_data, epochs, patience, batch_size):
        logging.info("Getting training data")

        if torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"

        device = torch.device(device_type)

        self.device = device

        logging.info(f"Predicting using device: {device_type}")
        early_stopping = EarlyStopping(patience=patience, delta=0.0001)

        # create dataloaders
        train_dataloader, data_shape = self.prepare_dataloader(
            data=train_data,
            batch_size=int(batch_size),
        )
        val_dataloader, _ = self.prepare_dataloader(
            data=validation_data,
            batch_size=int(batch_size),
        )

        ######################
        logging.info("Instantiating Neural CDE")

        model = NeuralCDE(
            input_channels_x=data_shape[2],
            hidden_channels_x=self.hidden_channels_x,
            output_channels=59,
        )

        model = model.to(self.device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

        self.run.watch(model, log="all")

        logging.info("Training CDE")
        epochs = 100

        # allows lambda to vary across training
        lambda_vals = np.linspace(0.0, 1.0, num=epochs)

        for epoch in tqdm(range(epochs)):

            lambda_val = lambda_vals[epoch]
            logging.info(f"Training epoch: {epoch} & lambda: {lambda_val}")
            # put into train mode
            model, train_losses_total, train_losses_y, train_losses_a = self._train(
                train_dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                lambda_val=lambda_val,
            )

            # zero grad check out + early stopping
            logging.info(f"Validation epoch: {epoch}")
            model, val_losses_total, val_losses_y, val_losses_a = self._test(
                test_dataloader=val_dataloader,
                model=model,
                lambda_val=lambda_val,
            )

            tqdm.write(
                f"Epoch: {epoch}   Training loss: {np.average(train_losses_total)} ; Train Treatment loss: {np.average(train_losses_a)} ; Train Outcome loss: {np.average(train_losses_y)}, Val loss: {np.average(val_losses_total)} ;Val Treatment loss: {np.average(val_losses_a)} ; Val Outcome loss: {np.average(val_losses_y)}",
            )

            if int(np.average(train_losses_y)) > 100000 or np.average(
                train_losses_y,
            ) == float("nan"):
                import sys

                raise ValueError("Exiting run...")

            self.run.log(
                {
                    "Epoch": epoch,
                    "Training loss": np.average(train_losses_total),
                    "Train Treatment loss": np.average(train_losses_a),
                    "Train Outcome loss": np.average(train_losses_y),
                    "Val loss": np.average(val_losses_total),
                    "Val Treatment loss": np.average(val_losses_a),
                    "Val Outcome loss": np.average(val_losses_y),
                },
            )

            # Log model checkpoint to wandb
            torch.save(model.state_dict(), f"./tmp_models/model_epoch_{epoch}.h5")
            self.run.save(f"./tmp_models/model_epoch_{epoch}.h5")

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(np.average(val_losses_y), model)

            if early_stopping.early_stop:
                print("Early stopping phase initiated...")
                break

        # load best model
        model.load_state_dict(torch.load("checkpoint.pt"))

        # Log model checkpoint to wandb
        torch.save(model.state_dict(), "./tmp_models/model_final.h5")
        self.run.save("./tmp_models/model_final.h5")

        self.model = model

    def predict(self, test_data):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Predicting with: {device}")

        treatment_loss = nn.CrossEntropyLoss()  # nn.NLLLoss()

        # training_processed
        (
            test_X,
            test_A,
            test_Time,
            test_y,
            test_treat,
            treated_indices,
            untreated_indices,
        ) = data_to_torch_tensor(
            test_data,
            sample_prop=self.sample_proportion,
        )

        test_concat = torch.cat((test_X, test_A), 2)

        # to make it continuous we interpolate (linear - keep causal interpretation)
        test_coeffs = torchcde.linear_interpolation_coeffs(test_concat)

        test_coeffs = torch.tensor(test_coeffs, dtype=torch.float, device=device)

        outcomes_test = torch.tensor(test_y[:, :, 0], dtype=torch.float, device=device)
        active_entries_test = torch.tensor(
            test_y[:, :, 1],
            dtype=torch.float,
            device=device,
        )
        current_treatment_test = torch.tensor(
            test_treat[:, -1, :],
            dtype=torch.float,
            device=device,
        )

        mcd = True
        if mcd:
            self.model = enable_dropout(self.model)

            pred_y_test_list = []
            pred_a_test_list = []
            z_hat_test_list = []
            pred_a_list = []

            for i in range(2):
                (
                    pred_y_test_sample,
                    pred_a_test_sample,
                    pred_a,
                    z_hat_sample,
                ) = self.model(test_coeffs, device)
                pred_y_test_list.append(pred_y_test_sample)
                pred_a_test_list.append(pred_a_test_sample)
                pred_a_list.append(pred_a)
                z_hat_test_list.append(z_hat_sample)

            write_to_file(pred_y_test_list, "test_y_preds.p")
            write_to_file(pred_a_test_list, "test_a_preds.p")
            write_to_file(z_hat_test_list, "latent.p")

            pred_y_test = torch.mean(torch.stack(pred_y_test_list), dim=0)
            pred_a_test = torch.mean(torch.stack(pred_a_test_list), dim=0)
            pred_a_out = torch.mean(torch.stack(pred_a_list), dim=0)

        else:
            pred_y_test, pred_a_test = self.model(test_coeffs, device)

        # compute norm mse loss - outcomes loss
        loss_y_test = compute_norm_mse_loss(
            outcomes_test,
            pred_y_test,
            active_entries_test,
        )

        loss_y_treated = compute_norm_mse_loss(
            outcomes_test[treated_indices],
            pred_y_test[treated_indices],
            active_entries_test[treated_indices],
        )

        loss_y_untreated = compute_norm_mse_loss(
            outcomes_test[untreated_indices],
            pred_y_test[untreated_indices],
            active_entries_test[untreated_indices],
        )

        # compute interventions/actions loss
        loss_a_test = treatment_loss(
            pred_a_test,
            torch.max(current_treatment_test, 1)[1],
        )

        acc = torchmetrics.functional.accuracy(
            F.softmax(pred_a_test, dim=1),
            torch.argmax(current_treatment_test, dim=1),
        )

        write_to_file(outcomes_test, "test_y.p")
        write_to_file(active_entries_test, "active.p")
        write_to_file(current_treatment_test, "test_a.p")

        self.run.log(
            {
                "Test Treatment loss": loss_a_test.item(),
                "RMSE Test Outcome loss": np.sqrt(loss_y_test.item()),
                "RMSE Test TREATED loss": np.sqrt(loss_y_treated.item()),
                "RMSE Test UNTREATED loss": np.sqrt(loss_y_untreated.item()),
                "Test ACC": acc.item(),
            },
        )

    def fit_multistep(
        self,
        train_data,
        validation_data,
        epochs,
        patience,
        batch_size,
        max_horizon,
    ):
        logging.info("Getting training data for multistep")

        if torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"

        device = torch.device(device_type)

        self.device = device

        logging.info(f"Predicting using device: {device_type}")
        early_stopping = EarlyStopping(patience=patience, delta=0.0001)

        # create dataloaders
        train_dataloader, data_shape = self.prepare_dataloader_multistep(
            data=train_data,
            batch_size=int(batch_size),
            max_horizon=max_horizon,
        )
        val_dataloader, _ = self.prepare_dataloader_multistep(
            data=validation_data,
            batch_size=int(batch_size),
            max_horizon=max_horizon,
        )

        ######################
        logging.info("Instantiating Neural CDE multistep")

        model_multistep = NeuralCDE(
            input_channels_x=data_shape[2],
            hidden_channels_x=self.hidden_channels_x,
            output_channels=max_horizon,
        )

        model_multistep = model_multistep.to(self.device)

        optimizer_multistep = torch.optim.SGD(
            model_multistep.parameters(),
            lr=0.0001,
            momentum=0.9,
        )
        # optimizer =  = torch.optim.Adam(model.parameters())  # , lr=0.01
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # , lr=0.01

        self.run.watch(model_multistep, log="all")

        logging.info("Training CDE")

        epochs = 10
        lambda_vals = np.linspace(0.0, 1.0, num=epochs)

        for epoch in tqdm(range(epochs)):

            lambda_val = lambda_vals[epoch]
            logging.info(f"Training epoch: {epoch} & lambda: {lambda_val}")
            # put into train mode

            (
                model_multistep,
                train_losses_total,
                train_losses_y,
                train_losses_a,
            ) = self._train(
                train_dataloader=train_dataloader,
                model=model_multistep,
                optimizer=optimizer_multistep,
                lambda_val=lambda_val,
            )

            # zero grad check out + early stopping
            logging.info(f"Validation epoch: {epoch}")
            model_multistep, val_losses_total, val_losses_y, val_losses_a = self._test(
                test_dataloader=val_dataloader,
                model=model_multistep,
                lambda_val=lambda_val,
            )

            tqdm.write(
                f"Epoch: {epoch}   Training loss: {np.average(train_losses_total)} ; Train Treatment loss: {np.average(train_losses_a)} ; Train Outcome loss: {np.average(train_losses_y)}, Val loss: {np.average(val_losses_total)} ;Val Treatment loss: {np.average(val_losses_a)} ; Val Outcome loss: {np.average(val_losses_y)}",
            )

            if int(np.average(train_losses_y)) > 100000 or np.average(
                train_losses_y,
            ) == float("nan"):
                import sys

                raise ValueError("Exiting multistep run...")

            self.run.log(
                {
                    "Epoch": epoch,
                    "Training loss": np.average(train_losses_total),
                    "Train Treatment loss": np.average(train_losses_a),
                    "Train Outcome loss": np.average(train_losses_y),
                    "Val loss": np.average(val_losses_total),
                    "Val Treatment loss": np.average(val_losses_a),
                    "Val Outcome loss": np.average(val_losses_y),
                },
            )

            # Log model checkpoint to wandb
            torch.save(
                model_multistep.state_dict(),
                f"./tmp_models/model_multi_epoch_{epoch}.h5",
            )
            self.run.save(f"./tmp_models/model_multi_epoch_{epoch}.h5")

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(np.average(val_losses_y), model_multistep)

            if early_stopping.early_stop:
                print("Early stopping phase initiated...")
                break

        # load best model
        model_multistep.load_state_dict(torch.load("checkpoint.pt"))

        # Log model checkpoint to wandb
        torch.save(model_multistep.state_dict(), "./tmp_models/model_multi_final.h5")
        self.run.save("./tmp_models/model_multi_final.h5")

        self.multistep_model = model_multistep

    def multistep_predict(self, test_data, max_horizon):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Predicting with: {device}")

        treatment_loss = nn.CrossEntropyLoss()  # nn.NLLLoss()

        if torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"

        device = torch.device(device_type)

        test_X, test_A, test_Time, test_y, test_treat, _, _ = data_to_torch_tensor(
            test_data,
            sample_prop=self.sample_proportion,
        )

        encode_concat = torch.cat((test_X, test_A), 2)

        # to make it continuous we interpolate (linear - keep causal interpretation)
        encode_coeffs = torchcde.linear_interpolation_coeffs(encode_concat)

        encode_coeffs = torch.tensor(encode_coeffs, dtype=torch.float, device=device)

        pred_y_test_sample, pred_a_test_sample, pred_a_test, z_hat_sample = self.model(
            encode_coeffs,
            device,
        )

        logging.info("Processing counterfactual data...")
        processed_data_multi = process_counterfactual_seq_test_data(
            data_map=test_data,
            states=z_hat_sample.cpu().detach().numpy(),
            projection_horizon=max_horizon,
        )

        test_X, test_A, test_Time, test_y, test_treat = data_to_torch_tensor_multistep(
            processed_data_multi,
            sample_prop=self.sample_proportion,
            max_horizon=max_horizon,
        )

        data_concat = torch.cat((test_X, test_A), 2)

        data_shape = list(data_concat.shape)

        # to make it continuous we interpolate (linear - keep causal interpretation)
        test_coeffs = torchcde.linear_interpolation_coeffs(data_concat)

        test_coeffs = torch.tensor(test_coeffs, dtype=torch.float, device=device)

        outcomes_test = torch.tensor(test_y[:, :, 0], dtype=torch.float, device=device)
        active_entries_test = torch.tensor(
            test_y[:, :, 1],
            dtype=torch.float,
            device=device,
        )
        current_treatment_test = torch.tensor(
            test_treat[:, -1, :],
            dtype=torch.float,
            device=device,
        )

        pred_y_test, pred_a_test, pred_a = self.pred_auto(
            test_coeffs,
            device=device,
            max_horizon=max_horizon,
        )

        loss_y_test = compute_norm_mse_loss(
            outcomes_test[:, max_horizon - 1],
            pred_y_test[:, max_horizon - 1],
            active_entries_test[:, max_horizon - 1],
        )

        rmses = []
        accs = []

        rmses.append(
            torch.sqrt(
                compute_norm_mse_loss(outcomes_test, pred_y_test, active_entries_test),
            ).item(),
        )

        for i in range(max_horizon):
            rmses.append(
                torch.sqrt(
                    compute_norm_mse_loss(
                        outcomes_test[:, i],
                        pred_y_test[:, i],
                        active_entries_test[:, i],
                    ),
                ).item(),
            )

        acc_scores = []
        for i in range(max_horizon):
            true = torch.tensor(
                processed_data_multi["current_treatments"][:, i, :],
                dtype=torch.float,
                device=device,
            )
            acc_scores.append(
                torchmetrics.functional.accuracy(
                    F.softmax(pred_a[i], dim=1),
                    torch.argmax(true, dim=1),
                ),
            )

        self.run.log(
            {
                "RMSE MULTISTEP Test Outcome loss": np.sqrt(loss_y_test.item()),
            },
        )

        # RMSEs at other interval
        for i in range(len(rmses) - 1):
            self.run.log({f"RMSE @ {i+1} ": rmses[::-1][i]})

        for idx, acc in enumerate(acc_scores):
            self.run.log(
                {
                    f"MULTISTEP ACC @ {idx+1}": acc.item(),
                },
            )

    def pred_auto(self, test_coeffs, device, max_horizon):
        projection_horizon = max_horizon
        preds = []
        pred_a = []
        for t in range(0, projection_horizon):
            predictions, pred_a_sample_test, pred_a_sample, _ = self.multistep_model(
                test_coeffs,
                device,
            )
            preds.append(predictions)
            pred_a.append(pred_a_sample)
        return predictions, pred_a_sample_test, pred_a

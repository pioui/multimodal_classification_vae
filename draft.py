            epoch_dict["overall_val_loss":torch.mean(torch.tensor(overall_val_loss_list)) ]
            epoch_dict["θ_val_loss":torch.mean(torch.tensor(theta_val_loss_list)) ]
            epoch_dict["φ_val_loss":torch.mean(torch.tensor(psi_val_loss_list)) ]
            val_loss_dict.append(epoch_dict)

            logger.info(
                "Epoch {} Training: val_loss = {}, θ_val_loss = {}, φ_val_loss {}".format(
                    epoch,
                    epoch_dict["overall_val_loss"], 
                    epoch_dict["θ_val_loss"], 
                    epoch_dict["φ_val_loss"], 
                    )
                    )
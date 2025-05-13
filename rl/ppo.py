import torch
import torch.nn as nn
import torch.optim as optim
from vae_pretrain import KL_loss, MSE, reconstruction_loss

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 PaS_coef = None,
                 PaS_est_coef = None, 
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.PaS_coef = PaS_coef
        self.PaS_est_coef = PaS_est_coef 

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    
        if actor_critic.config.pas.encoder_type == 'vae':
            if actor_critic.config.pas.PaS_coef > 0.:
                # Freeze Label_VAE during the training
                for param in self.actor_critic.base.Label_VAE.parameters():
                    param.requires_grad = False                 

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)
    

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        k_loss_epoch = 0
        PaS_loss_epoch = 0
        PaS_est_loss_epoch = 0


        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample


                values, action_log_probs, dist_entropy, _, z_l, z, decoded, mu, logvar = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()                

                self.optimizer.zero_grad()
                
                total_loss=value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                
                if self.actor_critic.config.pas.encoder_type =='vae':
                    k_loss = KL_loss(mu, logvar)   
                    total_loss += k_loss * 0.5
                    
                    ## Raise error if both PaS_coef and PaS_est_coef are not set
                    if self.PaS_coef is [None, 0.] and self.PaS_est_coef is [None, 0.]:
                        raise ValueError('Both PaS_coef and PaS_est_coef are not set. Please set at least one of them.')
                    
                    if self.PaS_coef > 0.:
                        PaS_loss = MSE(z, z_l)
                        total_loss += PaS_loss * self.PaS_coef
                    else:
                        PaS_loss = torch.tensor(0.0).cuda()
                                                
                    if self.PaS_est_coef >0.: 
                        import pdb
                        
                        ## Estimating only occluded human agents
                        # Remove walls
                        target_grid = obs_batch['label_grid'][:,0].clone()
                        target_grid[obs_batch['label_grid'][:,1]==-9999.] = 0.0
                        # Remove seen (including partially visible) humans
                        target_grid[obs_batch['grid'][:,-1]==1.] == 0.0              

                        PaS_est_loss = reconstruction_loss(decoded.squeeze(1), target_grid)
                        total_loss += PaS_est_loss * self.PaS_est_coef
                    else:
                        PaS_est_loss = torch.tensor(0.0).cuda()
                else:
                    k_loss = torch.tensor(0.0).cuda()
                    PaS_loss = torch.tensor(0.0).cuda()
                    PaS_est_loss = torch.tensor(0.0).cuda()                       

                total_loss.backward()
                
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)

                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()  
                k_loss_epoch += k_loss.item()
                PaS_loss_epoch += PaS_loss.item()
                PaS_est_loss_epoch += PaS_est_loss.item()                

                # if self.actor_critic.config.pas.encoder_type =='vae' self.actor_critic.config.pas.PaS_coef > 0.: 
                #     PaS_loss_epoch += PaS_loss.item()   
                #     if self.PaS_est_coef >0.:     
                #         PaS_est_loss_epoch += est_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        k_loss_epoch /= num_updates
        PaS_loss_epoch /= num_updates
        PaS_est_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, k_loss_epoch, PaS_loss_epoch, PaS_est_loss_epoch

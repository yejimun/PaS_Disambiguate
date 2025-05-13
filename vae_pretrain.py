import os
import matplotlib.pyplot as plt
import glob 
import numpy as np
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from arguments import get_args
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from rl.pas_rnn_model import Label_VAE, Sensor_VAE
from crowd_nav.configs.config import Config
from crowd_sim.envs.grid_utils import MapSimilarityMetric
import pdb

# TODO:
# 1. Ablation study between PaS only and PaS_est only
# 2. Ablation study betweeen MSE loss and BCE loss




def make_sequence(data_path, phase, sequence=1):    
    label_grid_files = sorted(glob.glob(data_path+'*label_grid.npy'))
    sensor_grid_files = sorted(glob.glob(data_path+'*sensor_grid.npy'))
    id_grid_files = sorted(glob.glob(data_path+'*id_grid.npy'))
    if phase !='train':
        vector_files = sorted(glob.glob(data_path+'*vector.npy'))

    vector = []
    label_grid = []
    sensor_grid = []
    id_grid = []
    for i in range(len(label_grid_files)):
        if phase != 'train':
            v_f, lg_f, sg_f, id_f = vector_files[i], label_grid_files[i], sensor_grid_files[i], id_grid_files[i] 
        else:
            lg_f, sg_f, id_f = label_grid_files[i], sensor_grid_files[i], id_grid_files[i] 
        
        epi_label_grid = np.load(lg_f, mmap_mode='r')
        epi_sensor_grid = np.load(sg_f, mmap_mode='r')
        epi_id_grid = np.load(id_f, mmap_mode='r')
        if phase != 'train':
            epi_vector = np.load(v_f, mmap_mode='r')
        timestamp = 0
        for k in range(len(epi_label_grid)):   
            if phase == 'train':  
                lg, sg, ig  = epi_label_grid[k], epi_sensor_grid[k], epi_id_grid[k]
            else:
                v, lg,sg, ig = epi_vector[k], epi_label_grid[k], epi_sensor_grid[k], epi_id_grid[k]
            if timestamp == 0.:
                lg = lg # (100, 100)
                sg = sg
                ig = ig
                if phase != 'train':
                    v = v

                labelG = deque(maxlen=1)
                sensorG = deque(maxlen=sequence)
                idG = deque(maxlen=1)
                if phase != 'train':
                    vec = deque(maxlen=sequence)

                sensorG.extend(np.vstack([sg for i in range(sequence)]))
                if phase != 'train':
                    vec.extend(np.vstack([v for i in range(sequence)])) 

            else:
                sensorG.extend(sg)
                if phase != 'train':
                    vec.extend(v)
            
            labelG.extend(lg)
            idG.extend(ig)
            
            label_grid.append(np.array(labelG, dtype=np.float32).copy())
            sensor_grid.append(np.array(sensorG, dtype=np.float32).copy())
            id_grid.append(np.array(idG, dtype=np.float32).copy())
            if phase != 'train':
                vector.append(np.array(vec, dtype=np.float32).copy())
            
            timestamp+=1
            
        if i % 50 == 0:
            print(i, 'file sequence has been made.')
            
    if phase == 'train':
        return label_grid, sensor_grid, id_grid
    else:        
        return vector, label_grid, sensor_grid, id_grid 




class DATA(Dataset):
    def __init__(self,logging, phase, sequence=1):
        data_path = 'entering_room_data/'+phase +'/' 
        self.phase = phase

        if phase =='train':
            self.label_grid, self.sensor_grid, self.id_grid = make_sequence(data_path, phase, sequence=sequence)
        else:
            self.vector, self.label_grid, self.sensor_grid, self.id_grid = make_sequence(data_path, phase, sequence=sequence)
        logging.info('Phase : {}, sequential data : {:d}'. format(phase, len(self.label_grid)))   

    def __len__(self):
        return len(self.label_grid)

    def __getitem__(self, index):
        label_grid = self.label_grid[index].copy()
        sensor_grid = self.sensor_grid[index].copy()
        id_grid = self.id_grid[index].copy()
        if self.phase != 'train':
            vector = self.vector[index].copy()
        #     id_grid = self.id_grid[index].copy()
            return (vector, label_grid, sensor_grid, id_grid)
        else:
        #     return (label_grid, sensor_grid)
            return (label_grid, sensor_grid, id_grid)

def _flatten_helper(T, N, _tensor):
    return _tensor.reshape(T * N, *_tensor.size()[2:])

def rearrange(data):
    # after loaded from the dataloader
    # (N, T, ~) --> (T, N, ~) --> (T*N, ~)
    data = data.unsqueeze(2).to(device)
    N = data.size()[0]
    T = data.size()[1]
    data = data.transpose(1,0)   
    return _flatten_helper(T, N, data.squeeze(1)) 

def stack_tensors(label_grid, sensor_grid, id_grid=None):
    label_grid = rearrange(label_grid)
    sensor_grid = rearrange(sensor_grid)
    if id_grid is not None:
        id_grid = rearrange(id_grid)
    # if mask is not None:
    #     mask = rearrange(mask)
    return label_grid, sensor_grid, id_grid# ,  mask

def to_cuda(label_grid, sensor_grid, id_grid=None): 
    if id_grid is not None:
        return label_grid.to(device), sensor_grid.to(device), id_grid.to(device)
    else:
        return label_grid.to(device), sensor_grid.to(device), id_grid


def reconstruction_loss(x, target):
    """[summary]

    Args:
        x ([N,H,W]): [description]
        x_recon ([N,H,W]): [description]
        distribution (str, optional): [description]. Defaults to 'gaussian'.
        overest (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    batch_size = x.size(0)
    H = x.size(-2)
    W = x.size(-1)
    assert batch_size != 0
    x = x.view(batch_size, H, W)
    target = target.view(batch_size, H, W)
    
    recon_loss = F.mse_loss(x,target, reduction='none').sum(dim=(1,2)).mean()
    return recon_loss


def MSE(x, target):
    """[summary]
    x: (N, 1, len)
    target: (N, 1, len)
    """
    batch_size = x.size(0)
    x = x.view(batch_size, -1)
    target = target.view(batch_size, -1)
    return F.mse_loss(x, target, reduction='none').sum(dim=1).mean()

def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0

def KL_loss(mu, logvar):
    """
    mu: (N, 128)
    logvar: (N, 128)
    """
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0) 
    return KLD


def VAE_loss(x, target, mu, logvar):
    MSE = reconstruction_loss(x, target)
    KLD = KL_loss(mu, logvar)
    return MSE, KLD


def Label_vae_evaluate(beta, logging, loader, model, epoch=None):
    recon_loss_epoch = []
    k_loss_epoch = []

    model.eval()

    with torch.no_grad():
        for vector, label_grid, sensor_grid, id_grid in loader:
            """
            vector : (T*N, 36)
            *_grid : (T*N, 120, 120)
            """     
            label_grid, sensor_grid, id_grid = stack_tensors(label_grid, sensor_grid, id_grid)  # (N,T, ...) --> (T*N, ...)                   

            mu_l, logvar_l, z_l, decoded_l = model(label_grid)    
            
            ## Estimating only humans
            # Remove walls
            target_grid = label_grid.clone()
            target_grid[id_grid==-9999.] = 0.0           
            
            # recon_loss, k_loss = VAE_loss(decoded_l, label_grid, mu_l, logvar_l)
            recon_loss, k_loss = VAE_loss(decoded_l, target_grid, mu_l, logvar_l)
            loss = recon_loss + k_loss * beta


            recon_loss_epoch.append(recon_loss.item())  
            k_loss_epoch.append(k_loss.item())            
        
        avg_recon_loss = average(recon_loss_epoch)
        avg_k_loss = average(k_loss_epoch)
        
        if epoch == None:    
            loss = logging.info('(Test) recon_loss: {:.4f}, k_loss: {:.4f}'. format(avg_recon_loss, avg_k_loss))  
            save_path = out_dir+'/Label_VAE_test_sample/'
        else:
            loss = logging.info('(Eval Epoch {:d}) recon_loss: {:.4f}, k_loss: {:.4f}'. format(epoch, avg_recon_loss, avg_k_loss))   
            save_path = out_dir+'/Label_VAE_val_sample/'

            writer.add_scalar('Label_VAE_val_recon_loss', avg_recon_loss, epoch)
            writer.add_scalar('Label_VAE_val_k_loss', avg_k_loss, epoch)

            
        if not os.path.exists(save_path):
            os.makedirs(save_path) 


        vectors = vector.cpu().numpy()
        for k in range(batch_size):
            fig, axes = plt.subplots(ncols=3, figsize=(6*3+2,6))
            ax = axes[0]
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)            
            
            robot_traj = []
            a1_traj = []
            a2_traj = []
            # a3_traj = []
            # a4_traj = []
            # a5_traj = []
            # a6_traj = []
            # vector length : robot 9, human 5
            robot_traj.append(vectors[k, 0, :2]) 
            a1_traj.append(vectors[k, 0, 9:11]) 
            a2_traj.append(vectors[k, 0, 14:16]) 
            # a3_traj.append(vectors[k, 0, 19:21]) 
            # a4_traj.append(vectors[k, 0, 24:26]) 
            # a5_traj.append(vectors[k, 0, 29:31]) 
            # a6_traj.append(vectors[k, 0, 34:36])

            robot_traj = np.array(robot_traj) 
            a1_traj = np.array(a1_traj)
            a2_traj = np.array(a2_traj)
            # a3_traj = np.array(a3_traj)
            # a4_traj = np.array(a4_traj)
            # a5_traj = np.array(a5_traj)
            # a6_traj = np.array(a6_traj)
            ax.plot(robot_traj[:,0], robot_traj[:,1], 'r+')
            ax.plot(a1_traj[:,0], a1_traj[:,1], 'bo')
            ax.plot(a2_traj[:,0], a2_traj[:,1], 'go')
            # ax.plot(a3_traj[:,0], a3_traj[:,1], 'yo')
            # ax.plot(a4_traj[:,0], a4_traj[:,1], 'co')
            # ax.plot(a5_traj[:,0], a5_traj[:,1], 'mo')
            # ax.plot(a6_traj[:,0], a6_traj[:,1], 'ko')
            ax.set_title('traj')
            i = 2 
            # for grid in [label_grid[k], decoded_l[k]]:
            for grid in [target_grid[k], decoded_l[k]]:
                ax = axes[i-1]
                Con = ax.contourf(grid.squeeze(0).cpu().numpy(), cmap='binary', vmin = 0.0, vmax = 1.0)
                i+=1     
            fig.colorbar(Con, ax = axes.ravel().tolist())

            if epoch == None:
                plt.savefig(save_path + 'ex_'+str(k)+'.png')
            else:
                plt.savefig(save_path + 'epoch_'+str(epoch)+'_ex_'+str(k)+'.png')
            plt.close()
    return loss


def Label_vae_train(beta, logging, train_loader, validation_loader, model, ckpt_path, num_epochs, learning_rate = 0.001):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) 
    model.train()
    for epoch in range(1, num_epochs+1):
        recon_loss_epoch = []
        k_loss_epoch = []
        loop = tqdm(train_loader, total = len(train_loader), leave = True)
        
        for label_grid, sensor_grid, id_grid in loop: 

            label_grid, sensor_grid, id_grid = stack_tensors(label_grid, sensor_grid, id_grid)  # (N,T, ...) --> (T*N, ...)       

            mu_l, logvar_l, z_l, decoded_l = model(label_grid)    
            
            ## Estimating only humans
            # Remove walls
            target_grid = label_grid.clone()
            target_grid[id_grid==-9999.] = 0.0     
            
            # recon_loss, k_loss = VAE_loss(decoded_l, label_grid, mu_l, logvar_l)
            recon_loss, k_loss = VAE_loss(decoded_l, target_grid, mu_l, logvar_l)
            loss = recon_loss + k_loss * beta
        

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(recon_loss=recon_loss.item(), k_loss=k_loss.item())
            recon_loss_epoch.append(recon_loss.item())
            k_loss_epoch.append(k_loss.item())

        avg_recon_loss = average(recon_loss_epoch)
        avg_k_loss = average(k_loss_epoch)

        logging.info('(Epoch {:d}) recon_loss: {:.4f}, k_loss: {:.4f}'.
                     format(epoch, avg_recon_loss, avg_k_loss))      

        writer.add_scalar('Label_VAE_train_recon_loss', avg_recon_loss, epoch)
        writer.add_scalar('Label_VAE_train_val_k_loss', avg_k_loss, epoch)
        
        if epoch % 20 == 0 :
            loop.set_postfix(loss = Label_vae_evaluate(beta, logging, validation_loader, model, epoch))
            ckpt_file = os.path.join(ckpt_path, 'label_weight_'+str(epoch)+'.pth')
            torch.save(model.state_dict(), ckpt_file)
            model.train()
        
        
def Sensor_vae_evaluate(beta, logging, loader, gt_model, model, epoch=None, PaS_coef=1, est_coef=1):
    PaS_loss_epoch = []
    total_loss_epoch = []
    est_loss_epoch = []
    precision_epoch = []
    recall_epoch = []
    accuracy_epoch = []

    precision_unoccupied_epoch = []
    recall_unoccupied_epoch = []

    model.eval()

    total_similarity = []
    total_occupied_similarity = []
    total_free_similarity = []
    total_occluded_similarity = []
    total_base_similarity = []
    total_base_occupied_similarity = []
    total_base_free_similarity = []
    total_base_occluded_similarity = []
    

    vectors = []
    label_grids = []
    sensor_grids = []
    decodeds = []


    with torch.no_grad():
        num_vis = 0
        for vector, label_grid, sensor_grid, id_grid in loader:
            """[summary]
            vector : (N, T, 36)
            _grid : (N, T, 120, 120)
            rnn_ : (N, T, 128)
            mask : (N, 5)
            """            
            # label_grid, sensor_grid, _ = stack_tensors(label_grid, sensor_grid)  # (N,T, ...) --> (T*N, ...)       
            label_grid, sensor_grid, id_grid = to_cuda(label_grid, sensor_grid, id_grid)
            
            mu, logvar, z = model(sensor_grid)  
            _, _, z_l, _ = gt_model(label_grid)
            decoded = gt_model.decoder(z) # Needed for visualization even if not used for training.
    
            ## Estimating only occluded human agents
            # Remove walls
            target_grid = label_grid[:,-1].clone()
            target_grid[id_grid[:,-1]==-9999.] = 0.0
            # Remove seen (including partially visible) humans. Collected data considers partially visible humans as fully visible
            target_grid[sensor_grid[:,-1]==1.] == 0.0     
            
            loss = KL_loss(mu, logvar)
            if est_coef>0:   
                # est_loss, k_loss = VAE_loss(decoded.squeeze(1),, label_grid[:,-1].squeeze(1), mu, logvar)
                est_loss = reconstruction_loss(decoded.squeeze(1), target_grid.squeeze(1))
                loss += est_coef * est_loss
            else:
                est_loss = torch.zeros(1).cuda()
            
            if PaS_coef>0:
                PaS_loss = MSE(z, z_l)     
                loss +=  PaS_coef * PaS_loss  
            else:
                PaS_loss = torch.zeros(1).cuda()
                
            y_label = target_grid.squeeze(1).float()
            y_pred = decoded.squeeze(1)
            y_pred[y_pred>=0.6] = 1.0
            y_pred[y_pred<=0.4] = 0.0
            y_pred[torch.logical_and(y_pred!=1.0, y_pred!=0.0)] = 0.5
            # y_pred = torch.where(y_pred>=0.6, 1.0, y_pred)
            # y_pred = torch.where(y_pred<=0.4, 0.0, y_pred)
            # y_pred = torch.where(torch.logical_and(y_pred!=1.0, y_pred!=0.0), 0.5, y_pred).double()


            # change y_pred 
            label_pos = torch.sum(y_label == 1).item()
            label_neg = torch.sum(y_label == 0).item()
            pred_pos = torch.sum(y_pred==1).item()
            pred_neg = torch.sum(y_pred==0).item()
            true_pos =  torch.sum((y_pred == y_label)*(y_pred==1)).item()
            true_neg =  torch.sum((y_pred == y_label)*(y_pred==0)).item()
            
            recall_unoccupied = true_neg / label_neg if not label_neg == 0 else -1
            precision_unoccupied = true_neg / pred_neg if not pred_neg == 0 else -1
            recall = true_pos / label_pos if not label_pos == 0 else -1 # prevent div by 0 error
            precision = true_pos / pred_pos if not pred_pos == 0 else -1 # prevent div by 0 error
            accuracy = (true_pos + true_neg) / (label_pos + label_neg)

            # if epoch is None: # During test phase only
            #     for est, obs, gt in zip(decoded.squeeze(1).cpu().numpy(), sensor_grid[:, -1].cpu().numpy(), label_grid.squeeze(1).cpu().numpy()):     
            #         similarity, base_similarity = MapSimilarityMetric(est, obs, gt)
            #         total_similarity.append(similarity[0])
            #         total_occupied_similarity.append(similarity[1])
            #         total_free_similarity.append(similarity[2])
            #         total_occluded_similarity.append(similarity[3])
                    
            #         total_base_similarity.append(base_similarity[0])
            #         total_base_occupied_similarity.append(base_similarity[1])
            #         total_base_free_similarity.append(base_similarity[2])
            #         total_base_occluded_similarity.append(base_similarity[3])  
                    

            if precision != -1: precision_epoch.append(precision)  
            if recall != -1: recall_epoch.append(recall)
            if precision_unoccupied != -1: precision_unoccupied_epoch.append(precision_unoccupied)  
            if recall_unoccupied != -1: recall_unoccupied_epoch.append(recall_unoccupied)  
            accuracy_epoch.append(accuracy)
                                      
            PaS_loss_epoch.append(PaS_loss.item())
            est_loss_epoch.append(est_loss.item()) 
            total_loss_epoch.append(loss.item())
            
            avg_PaS_loss = average(PaS_loss_epoch)
            avg_est_loss = average(est_loss_epoch)
            avg_total_loss = average(total_loss_epoch)
            
            
            if num_vis*sequence <= 20:
                vectors.append(vector)
                label_grids.append(label_grid[:,-1]) 
                sensor_grids.append(sensor_grid)
                decodeds.append(decoded)
                num_vis +=1

        avg_accuracy = average(accuracy_epoch)
        avg_precision = average(precision_epoch)
        avg_recall = average(recall_epoch)
        avg_precision_unoccupied = average(precision_unoccupied_epoch)
        avg_recall_unoccupied = average(recall_unoccupied_epoch)
           
        if epoch == None:  
        #     avg_occupied_smiliarity = average(total_occupied_similarity)
        #     avg_free_similarity = average(total_free_similarity)
        #     avg_occluded_similarity  = average(total_occluded_similarity)
        #     avg_base_occupied_smiliarity = average(total_base_occupied_similarity)
        #     avg_base_free_similarity = average(total_base_free_similarity)
        #     avg_base_occluded_similarity  = average(total_base_occluded_similarity)
                
        #     avg_similarity = average(total_similarity)
        #     avg_base_similarity = average(total_base_similarity)        
            
            if PaS_coef>0:
                loss = logging.info('(Test) total_loss: {:.4f}, PaS_loss: {:.4f}, est_loss: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, precision(unoccupied): {:.4f}, recall(unoccupied): {:.4f}'.
                format(avg_total_loss, avg_PaS_loss, avg_est_loss, avg_accuracy, avg_precision, avg_recall, avg_precision_unoccupied, avg_recall_unoccupied))     
            else:
                loss = logging.info('(Test) total_loss: {:.4f}, est_loss: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, precision(unoccupied): {:.4f}, recall(unoccupied): {:.4f}'.
                format(avg_total_loss, avg_est_loss, avg_accuracy, avg_precision, avg_recall, avg_precision_unoccupied, avg_recall_unoccupied))   
        #     logging.info('(Test) similarity: {:.2f}, base_similarity: {:.2f}'. format(avg_similarity, avg_base_similarity))    
        #     logging.info(
        # ' occupied image similarity(pas/sensor): {:.3f}/{:.3f} and free image similarity(pas/sensor): {:.3f}/{:.3f} and occluded image similarity(pas/sensor): {:.3f}/{:.3f} '.
        #         format(avg_occupied_smiliarity, avg_base_occupied_smiliarity, avg_free_similarity, avg_base_free_similarity, avg_occluded_similarity, avg_base_occluded_similarity))


            save_path = out_dir+'/Sensor_VAE_woEstLoss_test_sample/'
        else:
            if PaS_coef>0:
                loss = logging.info('(Eval Epoch {:d}) total_loss: {:.4f}, PaS_loss: {:.4f}, est_loss: {:.4f}'. format(epoch, avg_total_loss, avg_PaS_loss, avg_est_loss))    
                writer.add_scalar('Sensor_VAE_val_total_loss', avg_total_loss, epoch)
                writer.add_scalar('Sensor_VAE_val_PaS_loss', avg_PaS_loss, epoch)
            else:
                loss = logging.info('(Eval Epoch {:d}) est_loss: {:.4f}'. format(epoch, avg_est_loss))    
            save_path = out_dir+'/Sensor_VAE_woEstLoss_val_sample/'

            
            writer.add_scalar('Sensor_VAE_val_est_loss', avg_est_loss, epoch)


        if not os.path.exists(save_path):
            os.makedirs(save_path)   
        vectors = torch.cat(vectors).cpu().numpy() # (N, T, 36)
        label_grids = torch.cat(label_grids) # (N, 1, 120, 120)
        sensor_grids = torch.cat(sensor_grids) # (N, sequence, 120, 120)
        decodeds = torch.cat(decodeds)


        for k in range(label_grids.shape[0]):
            fig, axes = plt.subplots(ncols=4, figsize=(6*4+2,6))
            ax = axes[0]
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)            
            
            robot_traj = []
            a1_traj = []
            a2_traj = []
            # a3_traj = []
            # a4_traj = []
            # a5_traj = []
            # a6_traj = []
            for i in range(sequence): # vector length : robot 9, human 5
                robot_traj.append(vectors[k, i, :2]) # (N, T, 36)
                a1_traj.append(vectors[k, i, 9:11]) # vector[k, i, 6:8]
                a2_traj.append(vectors[k, i, 14:16]) # vector[k, i, 11:13]
                # a3_traj.append(vectors[k, i, 19:21]) # vector[k, i, 6:8]
                # a4_traj.append(vectors[k, i, 24:26]) # vector[k, i, 11:13]
                # a5_traj.append(vectors[k, i, 29:31]) # vector[k, i, 6:8]
                # a6_traj.append(vectors[k, i, 34:36]) # vector[k, i, 11:13]
            robot_traj = np.array(robot_traj) # (sequence,2)
            a1_traj = np.array(a1_traj)# (sequence,2)
            a2_traj = np.array(a2_traj)# (sequence,2)
            # a3_traj = np.array(a3_traj)# (sequence,2)
            # a4_traj = np.array(a4_traj)# (sequence,2)
            # a5_traj = np.array(a5_traj)# (sequence,2)
            # a6_traj = np.array(a6_traj)# (sequence,2)
            ax.plot(robot_traj[:,0], robot_traj[:,1], 'r+')
            ax.plot(a1_traj[:,0], a1_traj[:,1], 'bo')
            ax.plot(a2_traj[:,0], a2_traj[:,1], 'go')
            # ax.plot(a3_traj[:,0], a3_traj[:,1], 'yo')
            # ax.plot(a4_traj[:,0], a4_traj[:,1], 'co')
            # ax.plot(a5_traj[:,0], a5_traj[:,1], 'mo')
            # ax.plot(a6_traj[:,0], a6_traj[:,1], 'ko')
            
            ax.set_title('traj')
            i = 2 
            for grid in [label_grids[k], sensor_grids[k], decodeds[k]]:
                ax = axes[i-1] #ax = fig.add_subplot(1,4,i)
                if i == 3:
                    for seq in range(sequence-1):
                        Con = ax.contourf(grid[seq].squeeze(0).cpu().numpy(), cmap='binary', vmin = 0.0, vmax = 1.0, alpha=0.3)
                    Con = ax.contourf(grid[-1].squeeze(0).cpu().numpy(), cmap='binary', vmin = 0.0, vmax = 1.0)
                else:
                    Con = ax.contourf(grid.squeeze(0).cpu().numpy(), cmap='binary', vmin = 0.0, vmax = 1.0)
                i+=1     
            fig.colorbar(Con, ax = axes.ravel().tolist())
            # cbar_ticks = np.linspace(0., 1., num=5, endpoint=True)
            # cbar = plt.colorbar(Con,ticks=cbar_ticks)
            if epoch == None:
                plt.savefig(save_path + 'ex_'+str(k)+'.png')
            else:
                plt.savefig(save_path + 'epoch_'+str(epoch)+'_ex_'+str(k)+'.png')
            plt.close()   
    return loss


def Sensor_vae_train(beta, logging, train_loader, validation_loader, gt_model, model, ckpt_path, num_epochs, learning_rate = 0.001, PaS_coef=1, est_coef=1):
    ## Raise error if both PaS_coef and est_coef are not set
    if PaS_coef is [None, 0.] and est_coef is [None, 0.]:
        raise ValueError('Both PaS_coef and est_coef are not set. Please set at least one of them.')
    
    if gt_model is not None:
        for param in gt_model.parameters():
            param.requires_grad = False
    param_list = model.parameters()    

    PaS_optimizer = torch.optim.Adam(param_list, lr=learning_rate, weight_decay=1e-5) 
    model.train()
    if gt_model is not None:
        gt_model.train()
        
    for epoch in range(1, num_epochs+1):
        PaS_loss_epoch = []
        total_loss_epoch = []
        est_loss_epoch = []
        loop = tqdm(train_loader, total = len(train_loader), leave = True)
        
        for label_grid, sensor_grid, id_grid in loop:
            """[summary]
            vector : (N, T, 36)
            _grid : (N, T, 120, 120)
            rnn_ : (N, T, 128)
            mask : (N, 5)
            """            
            label_grid, sensor_grid, id_grid = to_cuda(label_grid, sensor_grid, id_grid)
            # label_grid, sensor_grid, _ = stack_tensors(label_grid, sensor_grid)  # (N,T, ...) --> (T*N, ...)       
            
            mu, logvar, z = model(sensor_grid)  
            with torch.no_grad():
                _, _, z_l, _ = gt_model(label_grid)
            
            
            loss = KL_loss(mu, logvar)
            if est_coef>0:
                decoded = gt_model.decoder(z)
                
                ## Estimating only occluded human agents
                # Remove walls
                target_grid = label_grid[:,-1].clone()
                target_grid[id_grid[:,-1]==-9999.] = 0.0
                # Remove seen (including partially visible) humans
                target_grid[sensor_grid[:,-1]==1.] == 0.0              
                
                # est_loss, k_loss = VAE_loss(decoded.squeeze(1), label_grid[:,-1].squeeze(1), mu, logvar)
                est_loss = reconstruction_loss(decoded.squeeze(1), target_grid.squeeze(1))
                loss += est_coef * est_loss 
            else:
                est_loss = torch.zeros(1).cuda()
                
            if PaS_coef>0:
                PaS_loss = MSE(z, z_l)    
                with torch.no_grad():
                    decoded = gt_model.decoder(z)
                loss +=  PaS_coef * PaS_loss          
            else:
                PaS_loss = torch.zeros(1).cuda()
                

            PaS_optimizer.zero_grad()
            loss.backward()
            PaS_optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(est_loss=est_loss.item(), PaS_loss=PaS_loss.item())

            PaS_loss_epoch.append(PaS_loss.item())
            est_loss_epoch.append(est_loss.item()) 
            total_loss_epoch.append(loss.item())
            
            
        avg_PaS_loss = average(PaS_loss_epoch)
        avg_est_loss = average(est_loss_epoch)
        avg_total_loss = average(total_loss_epoch)
        
        if est_coef > 0:     
            logging.info('(Epoch {:d}) total_loss: {:.4f}, PaS_loss: {:.4f}, est_loss: {:.4f}'.
                        format(epoch, avg_total_loss, avg_PaS_loss, avg_est_loss))   
            writer.add_scalar('Sensor_VAE_train_est_loss', avg_est_loss, epoch)
        else:
            logging.info('(Epoch {:d}) total_loss: {:.4f}, PaS_loss: {:.4f}'.
                        format(epoch, avg_total_loss, avg_PaS_loss))     
        writer.add_scalar('Sensor_VAE_train_PaS_loss', avg_PaS_loss, epoch)
        writer.add_scalar('Sensor_VAE_train_total_loss', avg_total_loss, epoch)
        
        if epoch % 20 == 0 :
            loop.set_postfix(loss = Sensor_vae_evaluate(beta, logging, validation_loader, gt_model, model, epoch, PaS_coef=PaS_coef, est_coef=est_coef))
            ckpt_file = os.path.join(ckpt_path, 'sensor_weight_'+str(epoch)+'.pth')
            torch.save(model.state_dict(), ckpt_file)
            model.train()
            if gt_model is not None:
                gt_model.train()






if __name__ == "__main__":

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 500 # 60 for the turtlebot experiment
    beta = 0.5

    algo_args = get_args()
    config = Config()
    encoder_type = 'vae'     
    sequence = config.pas.sequence
    label_vae_learning_rate =  0.001 
    sensor_vae_learning_rate = 0.001 
    PaS_coef = config.pas.PaS_coef
    est_coef = config.pas.est_coef
    sequence = config.pas.sequence 
    
    max_grad_norm = algo_args.max_grad_norm
    batch_size = 32    
    grid_shape = [100, 100] 
         
    
    # ###########################################
    # Label_AE training
    
    import logging

    output_path = 'entering_room' #'crossing_H12' #'LabelVAE_CircleFOV30'

    # configure logging
    out_dir = 'data/'+ output_path  
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_file = os.path.join(out_dir, 'label_vae.log')
    mode = 'a'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %%M:%S")

    summary_path = out_dir+'/runs_label_vae' 
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    writer = SummaryWriter(summary_path) 
                        

    logging.info('(config) encodertype:{},beta: {:.6f}, seq_length: {:d}, '. format(encoder_type, beta, sequence))    


    label_vae_ckpt_path = out_dir+'/label_vae_ckpt/'    

    logging.info('Learning rate:{:.6f}'. format(label_vae_learning_rate))   
    

    
    if not os.path.exists(label_vae_ckpt_path):
        os.makedirs(label_vae_ckpt_path)

    
    if encoder_type == 'vae':
        label_vae = Label_VAE(algo_args)
        
    label_vae.to(device)

    # ## loading checkpoint for label_vae_train
    # #  if resume:
    # label_vae_ckpt_file = os.path.join(label_vae_ckpt_path, 'label_weight_'+str(300)+'.pth')
    # label_vae.load_state_dict(torch.load(label_vae_ckpt_file))

    
    # train_set = DATA(logging, 'train', sequence=1) 
    # val_set = DATA(logging, 'val', sequence=1) 
    # train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size,num_workers=1,pin_memory=True, drop_last=True)
    # validation_loader = DataLoader(dataset=val_set, shuffle=True, batch_size=batch_size,num_workers=1, pin_memory=True, drop_last=True)

    # Label_vae_train(beta, logging, train_loader, validation_loader, label_vae, label_vae_ckpt_path, num_epochs, label_vae_learning_rate)


    # test_set = DATA(logging,'test', sequence=1)
    # test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size,num_workers=1, pin_memory=True, drop_last=True)  # batch_size=100
     
    # Label_vae_evaluate(beta, logging, test_loader, label_vae)

    ####################################
    # Sensor_VAE training
    
    log_file = os.path.join(out_dir, 'sensosr_vae.log')
    mode = 'a'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %%M:%S")

    summary_path = out_dir+'/runs_sensor_vae' 
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    writer = SummaryWriter(summary_path) 
    
    label_vae_ckpt_file = os.path.join(label_vae_ckpt_path, 'label_weight_'+str(num_epochs)+'.pth')
    label_vae.load_state_dict(torch.load(label_vae_ckpt_file))
    
    sensor_vae_ckpt_path = out_dir+'/sensor_vae_ckpt/'  
    if not os.path.exists(sensor_vae_ckpt_path):
        os.makedirs(sensor_vae_ckpt_path)
    logging.info('(Loss coefficients) m: {:.6f}, est: {:.6f}'. format(PaS_coef, est_coef))    

    
    train_set = DATA(logging, 'train', sequence=sequence) 
    val_set = DATA(logging, 'val', sequence=sequence) 
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size,num_workers=1,pin_memory=True, drop_last=True)
    validation_loader = DataLoader(dataset=val_set, shuffle=True, batch_size=batch_size,num_workers=1, pin_memory=True, drop_last=True)

    if encoder_type == 'vae':
        sensor_vae = Sensor_VAE(algo_args, config)        
    sensor_vae.to(device)    
    
    Sensor_vae_train(beta, logging, train_loader, validation_loader, label_vae, sensor_vae, sensor_vae_ckpt_path, num_epochs, sensor_vae_learning_rate, PaS_coef=PaS_coef, est_coef=est_coef)

    # # Load sensor vae model
    # label_vae_ckpt_file = os.path.join(label_vae_ckpt_path, 'label_weight_'+str(num_epochs)+'.pth')
    # label_vae.load_state_dict(torch.load(label_vae_ckpt_file))
    # if encoder_type == 'vae':
    #     sensor_vae = Sensor_VAE(algo_args, config)        
    # sensor_vae.to(device)    
    # sensor_vae_ckpt_path = out_dir+'/sensor_vae_woEstLoss_ckpt/'  
    # sensor_vae_ckpt_file = os.path.join(sensor_vae_ckpt_path, 'sensor_weight_'+str(300)+'.pth')
    # sensor_vae.load_state_dict(torch.load(sensor_vae_ckpt_file))
    

    test_set = DATA(logging,'test', sequence=sequence)
    test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size,num_workers=1, pin_memory=True, drop_last=True)  # batch_size=100
     
    Sensor_vae_evaluate(beta, logging, test_loader, label_vae, sensor_vae, PaS_coef=PaS_coef, est_coef=est_coef)

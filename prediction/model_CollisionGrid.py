import torch
import torch.nn as nn
from torch.autograd import Variable

class CollisionGridModel(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(CollisionGridModel, self).__init__()

        self.args = args
    
        # Store required sizes
        self.rnn_size = args.rnn_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.gru = args.gru
        self.d = args.device

        self.num_sector = args.num_sector
        self.num_TTC = len(args.TTC)
        self.embedding_size_action = args.embedding_size_action

        # The LSTM cell for pedestrians
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)
 
        if self.gru:
            self.cell = nn.GRUCell(2*self.embedding_size, self.rnn_size)

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.num_TTC*self.num_sector, self.embedding_size) # uncomment this (new - ped)
        
        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)


    def forward(self, *args):

        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''

        # Construct the output variable
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]

        if self.gru:
            cell_states = None

        mask = args[4]
        grids_TTC = args[5]

        numNodes = input_data.shape[1]
        timestep_length = input_data.shape[0] # 1 during test and train with no teacher forcing. seq_length when traininig with teacher forcing
        outputs = Variable(torch.zeros(timestep_length * numNodes, self.output_size)).to(self.d)
     
        # For each frame in the sequence
        for framenum,frame in enumerate(input_data):

            # Peds present in the current frame
            curr_mask = mask[framenum,:]
            if torch.sum(curr_mask) == 0:
                # If no peds, then go to the next frame
                continue

            # List of nodes
            list_of_nodes = []
            for i in range(curr_mask.shape[0]):
                if curr_mask[i] == 1:
                    list_of_nodes.append(i)

            corr_index = Variable((torch.LongTensor(list_of_nodes))).to(self.d)

            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes,:2] # Getting only the x and y of each pedestrian for the input.
           
            # Get the corresponding grid masks
            grid_current = grids[framenum].to(self.d)
            grid_TTC_current = grids_TTC[framenum].to(self.d)

            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)

            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0, corr_index)

            # getting the max value of each grid cell for each ego agent (max since we are using (TTC_threshod - TTC)
            # So bigger values are riskier to collide)
            social_tensor = grid_TTC_current.max(1)[0] # this should become of size: num_agent * num_sector

            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            # input_embedded = self.relu(self.input_embedding_layer(nodes_current))
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))
            # tensor_embedded = self.relu(self.tensor_embedding_layer(social_tensor))

            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)


            if not self.gru:
                # One-step of the LSTM
                h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
            else:
                h_nodes = self.cell(concat_embedded, (hidden_states_current))

            # Compute the output
            outputs[framenum*numNodes + corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes

        # Reshape outputs
        outputs_return = Variable(torch.zeros((timestep_length), numNodes, self.output_size)).to(self.d)
      
        for framenum in range(timestep_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states
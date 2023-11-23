import os
import sys
import torch
import modal
import argparse
from network import S4PRED
from modal import Image


stub = modal.Stub("modal_prediction",image=modal.Image.debian_slim().pip_install("torch"))

# dockerfile_image = Image.from_dockerfile("Dockerfile")

@stub.function(mounts=[modal.Mount.from_local_dir(os.getcwd(), remote_path="/root/ss_pred")])
#image=dockerfile_image

def sec_struct_pred(records=[]):

    if not records:
        print("No records")
        return

    def initialize_model():

        device = torch.device('cpu:0') 
        s4pred = S4PRED().to(device)

        s4pred.eval()
        s4pred.requires_grad=False

        # Loading model parameters 
        scriptdir = "/root/ss_pred/weights"
                                    
        weight_files=['/weights_1.pt',
                    '/weights_2.pt',
                    '/weights_3.pt',
                    '/weights_4.pt',
                    '/weights_5.pt']

        # Manually listing
        s4pred.model_1.load_state_dict(torch.load(scriptdir + weight_files[0], map_location=lambda storage, loc: storage))
        s4pred.model_2.load_state_dict(torch.load(scriptdir + weight_files[1], map_location=lambda storage, loc: storage))
        s4pred.model_3.load_state_dict(torch.load(scriptdir + weight_files[2], map_location=lambda storage, loc: storage))
        s4pred.model_4.load_state_dict(torch.load(scriptdir + weight_files[3], map_location=lambda storage, loc: storage))
        s4pred.model_5.load_state_dict(torch.load(scriptdir + weight_files[4], map_location=lambda storage, loc: storage))
        
        return s4pred


    # initialize the model
    s4pred = initialize_model()

    # Mapping function
    def aas2int(seq):
        aanumdict = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 
                'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16,
                'W':17, 'Y':18, 'V':19}
        return [aanumdict.get(res, 20) for res in seq]

    ind2char={0:"C", 1:"H", 2:"E"}
    ss_out = []

    for record in records:
        name, seq = record.split('\t')
        seq = str(seq)
        # Sanity checks for lowercase characters, gaps, etc.
        name = name.rstrip().upper().replace('-','')
        iseq = aas2int(seq)
        
        with torch.no_grad():
            device = torch.device('cpu:0')
            ss_conf=s4pred(torch.tensor([iseq]).to(device))
            ss=ss_conf.argmax(-1)
            ss=ss.cpu().numpy()
            ss_out.append(name+ "\t" + "".join(list(map(ind2char.get, ss))))
    return ss_out



@stub.local_entrypoint()
def main():
    # input_text_file = input("Enter the name of input text file:")
    # output_text_file = input("Enter the name of output text file:")

    batch_size = 30  
    batch_count = 10


    with open("sample_input.txt", 'r') as input_file:  #test_psipred_dataset
        with open("sample_output.txt", 'w') as output_file:
            while True:
                try:
                    batches = []
                    for _ in range(batch_count):
                        head = []
                        for _ in range(batch_size):
                            try:
                                head.append(next(input_file))
                            except StopIteration:
                                break
                        if head:
                            batches.append(head)
                        
                    if not batches:
                        break   # Empty batch
                    

                    for ret in sec_struct_pred.map(batches):
                        for item in ret:
                            output_file.write(item + "\n")
                            
                
                except StopIteration:
                    break   # EOF reached


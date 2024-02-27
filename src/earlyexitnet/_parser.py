import argparse
from earlyexitnet.tools import path_check
    
def args_parser():
    parser = argparse.ArgumentParser(description="Early Exit CLI")
    
    parser.add_argument('-m','--model_name',default="b_lenet_fcn",
            required=False, help='select the model name - see training model')

    parser.add_argument('-mp','--trained_model_path',metavar='PATH',type=path_check,
            required=False,
            help='Path to previously trained model to load, the same type as model name')

    parser.add_argument('-bstr','--batch_size_train',type=int,default=64,
                        help='batch size for the training of the network')
    parser.add_argument('-bbe','--bb_epochs', metavar='N',type=int, default=5, required=False,
            help='Epochs to train backbone(pretrain separately), or non ee network')
    parser.add_argument('-jte','--jt_epochs', metavar='n',type=int, default=5, required=False,
            help='after loading pretrained model(bbe training) epochs to train exits jointly with backbone')
    parser.add_argument('-exe','--ex_epochs', metavar='n',type=int, default=0, required=False,
            help='epochs to train exits with frozen backbone')
    parser.add_argument('-vf','--validation_frequency',type=int,default=1,required=False,
            help='Validation and save frequency. Number of epochs to wait for before valid,saving.')
    # opt selection
    parser.add_argument('-bbo','--bb_opt_cfg',type=str,default='adam-brn',required=False,
            help='Selection string to pick backbone optimiser configuration from training_tools')
    parser.add_argument('-jto','--jt_opt_cfg',type=str,default='adam-brn',required=False,
            help='Selection string to pick joint optimiser configuration from training_tools')
    parser.add_argument('-exo','--ex_opt_cfg',type=str,default='adam-brn',required=False,
            help='Selection string to pick exit-only optimiser configuration from training_tools')
    # run notes
    parser.add_argument('-rn', '--run_notes', type=str, required=False,
            help='Some notes to add to the train/test information about the model or otherwise')

    #parser.add_argument('--seed', metavar='N', type=int, default=random.randint(0,2**32-1),
    #    help='Seed for training, NOT CURRENTLY USED')

    parser.add_argument('-d','--dataset',
            choices=['mnist','cifar10','cifar100'],
            required=False, default='mnist',
            help='select the dataset, default is mnist')
    parser.add_argument('--no_scaling',action='store_true',
                        help='Prevents datqa being scaled to between 0,1')

    # choose the cuda device to target
    parser.add_argument('-gpu','--gpu_target',type=int,required=False,default=0,
            help="GPU acceleration target, int val for torch.device( cuda:[?] )")
    parser.add_argument('-nw','--num_workers',type=int,required=False,
            help="Number of workers for data loaders")

    #threshold inputs for TESTING
    parser.add_argument('-bste','--batch_size_test',type=int,default=1,
                        help='batch size for the testing of the network')
    #threshold inputs for testing, 1 or more args - user should know model
    parser.add_argument('-t1','--top1_threshold', nargs='+',type=float,required=False)
    parser.add_argument('-entr','--entr_threshold', nargs='+',type=float,required=False)

    # generate onnx graph for the model
    parser.add_argument('-go', '--generate_onnx',metavar='PATH',type=path_check,
                        required=False,
                        help='Generate onnx from loaded or trained Pytorch model, specify the directory of the output onnx')

    return parser



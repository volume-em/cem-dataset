import sys, os, yaml
import mlflow
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR, LambdaLR
from tqdm import tqdm

#relative imports
from metrics import ComposeMetrics, IoU, EMAMeter, AverageMeter

metric_lookup = {'IoU': IoU}

class DataFetcher:
    """
    Loads batches of images and masks from a dataloader onto the gpu.
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.loader_iter = iter(dataloader)

    def __len__(self):
        return len(self.dataloader)
    
    def reset_loader(self):
        self.loader_iter = iter(self.dataloader)

    def load(self):
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.reset_loader()
            batch = next(self.loader_iter)
            
        #get the images and masks as cuda float tensors
        images = batch['image'].float().cuda(non_blocking=True)
        masks = batch['mask'].cuda(non_blocking=True)
        return images, masks
        
class Trainer:
    """
    Handles model training and evaluation.
    
    Arguments:
    ----------
    config: A dictionary of training parameters, likely from a .yaml
    file
    
    model: A pytorch segmentation model (e.g. DeepLabV3)
    
    trn_data: A pytorch dataloader object that will return pairs of images and
    segmentation masks from a training dataset
    
    val_data: A pytorch dataloader object that will return pairs of images and
    segmentation masks from a validation dataset.
    
    """
    
    def __init__(self, config, model, trn_data, val_data=None):
        self.config = config
        self.model = model.cuda()
        self.trn_data = DataFetcher(trn_data)
        self.val_data = val_data
        
        #create the optimizer
        if config['optim'] == 'SGD':
            self.optimizer = SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['wd'])
        elif config['optim'] == 'AdamW':
            self.optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd']) #momentum is default
        else:
            optim = config['optim']
            raise Exception(f'Optimizer {optim} is not supported! Must be SGD or AdamW')
            
        #create the learning rate scheduler
        schedule = config['lr_policy']
        if schedule == 'OneCycle':
            self.scheduler = OneCycleLR(self.optimizer, config['lr'], total_steps=config['iters'])
        elif schedule == 'MultiStep':
            self.scheduler = MultiStepLR(self.optimizer, milestones=config['lr_decay_epochs'])
        elif schedule == 'Poly':
            func = lambda iteration: (1 - (iteration / config['iters'])) ** config['power']
            self.scheduler = LambdaLR(self.optimizer, func)
        else:
            lr_policy = config['lr_policy']
            raise Exception(f'Policy {lr_policy} is not supported! Must be OneCycle, MultiStep or Poly')        
            
        #create the loss criterion
        if config['num_classes'] > 1:
            #load class weights if they were given in the config file
            if 'class_weights' in config:
                weight = torch.Tensor(config['class_weights']).float().cuda()
            else:
                weight = None
                
            self.criterion = nn.CrossEntropyLoss(weight=weight).cuda()
        else:
            self.criterion = nn.BCEWithLogitsLoss().cuda()
        
        #define train and validation metrics and class names
        class_names = config['class_names']

        #make training metrics using the EMAMeter. this meter gives extra
        #weight to the most recent metric values calculated during training
        #this gives a better reflection of how well the model is performing
        #when the metrics are printed
        trn_md = {name: metric_lookup[name](EMAMeter()) for name in config['metrics']}
        self.trn_metrics = ComposeMetrics(trn_md, class_names)
        self.trn_loss_meter = EMAMeter()
        
        #the only difference between train and validation metrics
        #is that we use the AverageMeter. this is because there are
        #no weight updates during evaluation, so all batches should 
        #count equally
        val_md = {name: metric_lookup[name](AverageMeter()) for name in config['metrics']}
        self.val_metrics = ComposeMetrics(val_md, class_names)
        self.val_loss_meter = AverageMeter()
        
        self.logging = config['logging']
        
        #now, if we're resuming from a previous run we need to load
        #the state for the model, optimizer, and schedule and resume
        #the mlflow run (if there is one and we're using logging)
        if config['resume']:
            self.resume(config['resume'])
        elif self.logging:
            #if we're not resuming, but are logging, then we
            #need to setup mlflow with a new experiment
            #everytime that Trainer is instantiated we want to
            #end the current active run and let a new one begin
            mlflow.end_run()
            
            #extract the experiment name from config so that
            #we know where to save our files, if experiment name
            #already exists, we'll use it, otherwise we create a
            #new experiment
            mlflow.set_experiment(self.config['experiment_name'])

            #add the config file as an artifact
            mlflow.log_artifact(config['config_file'])
                
    def resume(self, checkpoint_fpath):
        """
        Sets model parameters, scheduler and optimizer states to the
        last recorded values in the given checkpoint file.
        """
        checkpoint = torch.load(checkpoint_fpath, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.logging and 'run_id' in checkpoint:
            mlflow.start_run(run_id=checkpoint['run_id'])
        
        print(f'Loaded state from {checkpoint_fpath}')
        print(f'Resuming from epoch {self.scheduler.last_epoch}...')

    def log_metrics(self, step, dataset):
        #get the corresponding losses and metrics dict for
        #either train or validation sets
        if dataset == 'train':
            losses = self.trn_loss_meter
            metric_dict = self.trn_metrics.metrics_dict
        elif dataset == 'valid':
            losses = self.val_loss_meter
            metric_dict = self.val_metrics.metrics_dict
            
        #log the last loss, using the dataset name as a prefix
        mlflow.log_metric(dataset + '_loss', losses.avg, step=step)
        
        #log all the metrics in our dict, using dataset as a prefix
        metrics = {}
        for k, v in metric_dict.items():
            values = v.meter.avg
            for class_name, val in zip(self.trn_metrics.class_names, values):
                metrics[dataset + '_' + class_name + '_' + k] = float(val.item())
                
        mlflow.log_metrics(metrics, step=step)
    
    def train(self):
        """
        Defines a pytorch style training loop for the model withtqdm progress bar
        for each epoch and handles printing loss/metrics at the end of each epoch.
        
        epochs: Number of epochs to train model
        train_iters_per_epoch: Number of training iterations is each epoch. Reducing this 
        number will give more frequent updates but result in slower training time.
        
        Results:
        ----------
        
        After train_iters_per_epoch iterations are completed, it will evaluate the model
        on val_data if there is any, then prints loss and metrics for train and validation
        datasets.
        """
        
        #set the inner and outer training loop as either 
        #iterations or epochs depending on our scheduler
        if self.config['lr_policy'] != 'MultiStep':
            last_epoch = self.scheduler.last_epoch + 1
            total_epochs = self.config['iters']
            iters_per_epoch = 1
            outer_loop = tqdm(range(last_epoch, total_epochs + 1), file=sys.stdout, initial=last_epoch, total=total_epochs)
            inner_loop = range(iters_per_epoch)
        else:
            last_epoch = self.scheduler.last_epoch + 1
            total_epochs = self.config['epochs']
            iters_per_epoch = len(self.trn_data)
            outer_loop = range(last_epoch, total_epochs + 1)
            inner_loop = tqdm(range(iters_per_epoch), file=sys.stdout)

        #determine the epochs at which to print results
        eval_epochs = total_epochs // self.config['num_prints']
        save_epochs = total_epochs // self.config['num_save_checkpoints']
        
        #perform training over the outer and inner loops
        for epoch in outer_loop:
            for iteration in inner_loop:
                #load the next batch of training data
                images, masks = self.trn_data.load()
                
                #run the training iteration
                loss, output = self._train_1_iteration(images, masks)
                
                #record the loss and evaluate metrics
                self.trn_loss_meter.update(loss)
                self.trn_metrics.evaluate(output.cpu(), masks.cpu())
                
            #when we're at an eval_epoch we want to print
            #the training results so far and then evaluate
            #the model on the validation data
            if epoch % eval_epochs == 0:
                #before printing results let's record everything in mlflow
                #(if we're using logging)
                if self.logging:
                    self.log_metrics(epoch, dataset='train')
                
                print('\n') #print a new line to give space from progess bar
                print(f'train_loss: {self.trn_loss_meter.avg:.3f}')
                self.trn_loss_meter.reset()
                #prints and automatically resets the metric averages to 0
                self.trn_metrics.print()
                
                #run evaluation if we have validation data
                if self.val_data is not None:
                    self.evaluate()
                    
                    if self.logging:
                        self.log_metrics(epoch, dataset='valid')

                    print('\n') #print a new line to give space from progess bar
                    print(f'valid_loss: {self.val_loss_meter.avg:.3f}')
                    self.val_loss_meter.reset()
                    #prints and automatically resets the metric averages to 0
                    self.val_metrics.print()
                    
            #update the optimizer schedule
            self.scheduler.step()
                    
            #the last step is to save the training state if 
            #at a checkpoint
            if epoch % save_epochs == 0:
                self.save_state(epoch)
                
                
    def _train_1_iteration(self, images, masks):
        #run a training step
        self.model.train()
        self.optimizer.zero_grad()

        #forward pass
        output = self.model(images)
        loss = self.criterion(output, masks)

        #backward pass
        loss.backward()
        self.optimizer.step()
        
        #return the loss value and the output
        return loss.item(), output.detach()

    def evaluate(self):
        """
        Evaluation method used at the end of each epoch. Not intended to
        generate predictions for validation dataset, it only returns average loss
        and stores metrics for validaiton dataset.
        
        Use Validator class for generating masks on a dataset.
        """
        #set the model into eval mode
        self.model.eval()
        
        val_iter = DataFetcher(self.val_data)
        for _ in range(len(val_iter)):
            with torch.no_grad():
                #load batch of data
                images, masks = val_iter.load()
                output = self.model.eval()(images)
                loss = self.criterion(output, masks)
                self.val_loss_meter.update(loss.item())
                self.val_metrics.evaluate(output.detach().cpu(), masks.cpu())
                
        #loss and metrics are updated inplace, so there's nothing to return
        return None
    
    def save_state(self, epoch):
        """
        Saves the self.model state dict
        
        Arguments:
        ------------
        
        save_path: Path of .pt file for saving
        
        Example:
        ----------
        
        trainer = Trainer(...)
        trainer.save_model(model_path + 'new_model.pt')
        """
        state = {'state_dict': self.model.state_dict(),
                 'scheduler': self.scheduler.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        
        if self.logging:
            state['run_id'] = mlflow.active_run().info.run_id
            
        #the last step is to create the name of the file to save
        #the format is: name-of-experiment_pretraining_epoch.pth
        model_dir = self.config['model_dir']
        exp_name = self.config['experiment_name']
        pretraining = self.config['pretraining']
        if os.path.isfile(pretraining):
            #this is slightly clunky, but it handles the case
            #of using custom pretrained weights from a file
            #usually there aren't any '.'s other than the file
            #extension
            pretraining = pretraining.split('/')[-1].split('.')[0]
            
        save_path = os.path.join(model_dir, f'{exp_name}_{pretraining}_epoch{epoch}.pth.tar')   
        torch.save(state, save_path)
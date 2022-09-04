import os
from matplotlib.pyplot import get
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from datasets.utils import get_train_loaders
from unet3d.losses import get_loss_criterion, get_losses_criterion, GANLoss, ContextuaLoss
from unet3d.metrics import get_evaluation_metric
from unet3d.model import get_model, define_G
from unet3d.utils import get_logger, get_tensorboard_formatter, create_optimizer, \
    create_lr_scheduler, get_number_of_learnable_parameters, ImagePool
from . import utils
import tqdm
import copy


logger = get_logger('UNet3DTrainer')


def create_ARNet_GAN_trainer(config):
    # Create the model
    # model = get_model(config['model'])
    model = define_G(**config['model'])
    # Create the refine model 
    refine_model = get_model(config['refine_model'])
    # Create the disciminitor 
    dis_model = get_model(config['discriminitor'])
    # use DataParallel if more than 1 GPU available
    device = torch.device(config['device'])
    # if torch.cuda.device_count() > 1 and not device.type == 'cpu':
    #     model = nn.DataParallel(model)
    #     logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

    # put the models on GPUs
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(device)
    refine_model = refine_model.to(device)
    dis_model = dis_model.to(device)
   
    
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params of preliminary model {get_number_of_learnable_parameters(model)}')
    logger.info(f'Number of learnable params o refine model {get_number_of_learnable_parameters(refine_model)}')


    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Creaye GAN loss criterion
    GAN_loss_criterion = GANLoss()
    # Create evaluation metrics
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer_pre = create_optimizer(config['optimizer'], model)
    optimizer_refine = create_optimizer(config['refine_optimizer'], refine_model)
    optimizer_disc = create_optimizer(config['optimizer'], dis_model)

    # Create learning rate adjustment strategy
    lr_config = config.get('lr_scheduler')
    lr_config_pre = copy.deepcopy(lr_config)
    # lr_scheduler_pre = create_lr_scheduler(config.get('lr_scheduler', None), optimizer_pre)
    # lr_scheduler_refine = create_lr_scheduler(None, optimizer_refine)
    lr_scheduler_pre = create_lr_scheduler(lr_config, optimizer_pre)
    lr_scheduler_refine = create_lr_scheduler(lr_config_pre, optimizer_refine)

    trainer_config = config['trainer']
    # Create tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)

    return UNet3DTrainer(
                         model=model,
                         refine_model=refine_model,
                         dis_model=dis_model,
                         optimizer_pre=optimizer_pre,
                         lr_scheduler_pre=lr_scheduler_pre,
                         optimizer_refine=optimizer_refine,
                         lr_scheduler_refine=lr_scheduler_refine,
                         optimizer_disc=optimizer_disc,
                         loss_criterion=loss_criterion,
                         GAN_loss_criterion=GAN_loss_criterion,
                         eval_criterion=eval_criterion,
                         tensorboard_formatter=tensorboard_formatter,
                         device=config['device'],
                         loaders=loaders,
                         resume=resume,
                         pre_trained=pre_trained,
                         **trainer_config)



class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, refine_model,dis_model, optimizer_pre, lr_scheduler_pre,
                 optimizer_refine, lr_scheduler_refine, optimizer_disc, loss_criterion,
                 GAN_loss_criterion, eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs, max_num_iterations,
                 validate_after_iters=200, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=False,
                 tensorboard_formatter=None, skip_train_validation=False,
                 resume=None, pre_trained=None, **kwargs):

        self.model = model
        self.refine_model = refine_model
        self.dis_model = dis_model
        self.optimizer_pre = optimizer_pre
        self.scheduler_pre = lr_scheduler_pre
        self.optimizer_refine = optimizer_refine
        self.scheduler_refine = lr_scheduler_refine
        self.optimizer_disc = optimizer_disc
        self.loss_criterion = loss_criterion
        self.GAN_loss_criterion = GAN_loss_criterion
        self.contextual_loss = ContextuaLoss()
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better

        logger.info(model)
        logger.info(refine_model)
        logger.info(dis_model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = utils.load_checkpoint(resume, self.model, self.optimizer_pre)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]

        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            utils.load_checkpoint(pre_trained, self.model, None)
            if 'checkpoint_dir' not in kwargs:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

    def fit(self):
        # for _ in range(self.num_epochs, self.max_num_epochs):
        epoch_list = range(0,self.max_num_epochs)
        for epoch in tqdm.tqdm(
                    enumerate(epoch_list), total=self.max_num_epochs,
                    desc='Train epoch==%d' % self.num_epochs, ncols=80,
                    leave=False):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()
        train_res_losses = utils.RunningAverage()
        train_pre_eval_scores = utils.RunningAverage()
        train_dis_losses = utils.RunningAverage()


        # sets the model in training mode
        self.model.train()
        self.refine_model.train()
        self.dis_model.train()
        for batch_idx, t in tqdm.tqdm(
                enumerate(self.loaders['train']), total=len(self.loaders['train']),
                desc='Train iteration=%d, in Epoch=%d' % (self.num_iterations,self.num_epochs), ncols=80, leave=False):
        # for t in self.loaders['train']:
        #     logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
        #                 f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}]')

            input, target, weight = self._split_training_batch(t)
            self.real_A = Variable(input)
            self.real_B = Variable(target)
            # residual = target-input
            self.real_residual = self.real_B - self.real_A
            self.fake_AB_pool = ImagePool(pool_size=4)

            # compute gradients and update parameters
            #update preliminary network first
            # self.optimizer_pre.zero_grad()
            # pre_output, loss = self._forward_pass(self.model, input, target, weight)
            # train_losses.update(loss.item(), self._batch_size(input))
            # loss.backward(retain_graph=True)
            # self.optimizer_pre.step()
            # #then update refine network
            # self.optimizer_refine.zero_grad()
            # preliminary_output = pre_output.detach()
            # pre_res = preliminary_output-input
            # estimated_res, loss_refine = self._forward_pass(self.refine_model, pre_res, residual, weight)
            # train_res_losses.update(loss_refine.item(), self._batch_size(input))
            # final_output = estimated_res+input
            # loss_refine.backward()
            # self.optimizer_refine.step()
            fake_B, _ = self._forward_G(is_training=True)

            self.optimizer_disc.zero_grad()
            loss_D = self._backward_D()
            train_dis_losses.update(loss_D.item(), self._batch_size(input))
            loss_D.backward()
            self.optimizer_disc.step()

            self.optimizer_pre.zero_grad()
            loss_G = self._backward_G()
            train_losses.update(loss_G.item(), self._batch_size(input))
            loss_G.backward()
            self.optimizer_pre.step()

            self.optimizer_refine.zero_grad()
            loss_AR, refine_residual = self._backward_AR()
            train_res_losses.update(loss_AR.item(), self._batch_size(input))
            loss_AR.backward()
            self.optimizer_refine.step()

            pre_output = fake_B
            final_output = refine_residual+self.real_A


            if self.num_iterations % self.validate_after_iters == 0:
                # set the models in eval mode
                self.model.eval()
                self.refine_model.eval()
                # evaluate on validation set
                eval_score, pre_eval_score, eval_loss, eval_res_loss = self.validate()
                # set the model back to training mode
                self.model.train()
                self.refine_model.train()

                # adjust learning rate if necessary
                if isinstance(self.scheduler_pre, ReduceLROnPlateau):
                    # self.scheduler_pre.step(eval_score)
                    self.scheduler_pre.step(pre_eval_score)
                else:
                    self.scheduler_pre.step()
                
                if isinstance(self.scheduler_refine, ReduceLROnPlateau):
                    self.scheduler_refine.step(eval_score)
                else:
                    self.scheduler_refine.step()
                
                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)
                # is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                # compute eval criterion
                if not self.skip_train_validation:
                    eval_score = self.eval_criterion(final_output, target)
                    pre_eval_score = self.eval_criterion(pre_output, target)
                    train_eval_scores.update(eval_score.item(), self._batch_size(input))
                    train_pre_eval_scores.update(pre_eval_score.item(), self._batch_size(input))

                # log stats, params and images
                logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Residual Loss: {train_res_losses.avg}. Discriminitor Loss: {train_dis_losses.avg}. Evaluation score: {train_eval_scores.avg}. Pre Evaluation score: {train_pre_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_res_losses.avg, train_dis_losses.avg, train_eval_scores.avg)
                self._log_params()
                self._log_images(input, target, final_output, pre_output, self.real_residual, refine_residual, 'train_')

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        # min_lr = 1e-7
        lr = self.optimizer_pre.param_groups[0]['lr']
        lr_refine = self.optimizer_refine.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True
        elif lr_refine < min_lr:
            logger.info(f'Refine Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()
        val_res_losses = utils.RunningAverage()
        val_pre_scores = utils.RunningAverage()
        val_iteration = 1
        with torch.no_grad():
            for batch_idx, t in tqdm.tqdm(
                enumerate(self.loaders['val']), total=len(self.loaders['val']),
                desc='Val iteration=%d' % (val_iteration), ncols=80, leave=False):
            # for i, t in enumerate(self.loaders['val']):
                # logger.info(f'Validation iteration {i}')
                val_iteration = val_iteration+1
                input, target, weight = self._split_training_batch(t)
                self.real_A = Variable(input)
                self.real_B = Variable(target)            
                self.real_residual = self.real_B - self.real_A
                self.fake_AB_pool = ImagePool(pool_size=4)
                #ARNetV2
                # residual = target-input
                # pre_output, loss = self._forward_pass(self.model,input, target, weight)
                # val_losses.update(loss.item(), self._batch_size(input))
                # preliminary_output = pre_output.detach()
                # pre_res = preliminary_output-input
                # estimated_res, loss_refine = self._forward_pass(self.refine_model, pre_res, residual, weight)
                # val_res_losses.update(loss_refine.item(), self._batch_size(input))
                # final_output = estimated_res+input

                # ARNet+GAN
                fake_B, loss = self._forward_G(is_training=False)
                val_losses.update(loss.item(), self._batch_size(input))
                loss_AR, refine_residual = self._backward_AR()
                val_res_losses.update(loss_AR.item(), self._batch_size(input))

                pre_output = fake_B
                final_output = refine_residual+self.real_A

                if batch_idx % 100 == 0:
                    self._log_images(input, target, final_output, pre_output, self.real_residual, refine_residual, 'val_')

                eval_score = self.eval_criterion(final_output, target)
                pre_eval_score = self.eval_criterion(pre_output, target)
                val_scores.update(eval_score.item(), self._batch_size(input))
                val_pre_scores.update(pre_eval_score.item(), self._batch_size(input))

                if self.validate_iters is not None and self.validate_iters <= batch_idx:
                    # stop validation
                    break

            self._log_stats('val', val_losses.avg, val_res_losses.avg, 0.0, val_scores.avg)
            logger.info(f'Validation finished. Loss: {val_losses.avg}. Residual Loss: {val_res_losses.avg}. Evaluation score: {val_scores.avg}. Pre Evaluation score: {val_pre_scores.avg}')
            return val_scores.avg, val_pre_scores.avg, val_losses.avg, val_res_losses.avg

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight

    def _forward_G(self, is_training):
        self.fake_B = self.model(self.real_A)
        self.pre_residual = self.fake_B.detach()-self.real_A
        self.refine_residual = self.refine_model(self.pre_residual)
        if is_training:
            loss = None
        else:
            loss = self.loss_criterion(self.fake_B, self.real_B)
        return self.fake_B, loss


    def _backward_AR(self):
        # self.pre_residual = self.fake_B.detach()-self.real_A
        # self.refine_residual = self.refine_model(self.pre_residual)
        self.loss_AR = self.loss_criterion(self.refine_residual, self.real_residual)
        return self.loss_AR, self.refine_residual
        
    def _backward_D(self):
        # Fake
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        # self.pred_fake = self.netD.forward(fake_AB.detach())
        self.pred_fake = self.dis_model(fake_AB.detach())
        pred_fake = self.pred_fake.cpu()
        self.loss_D_fake = self.GAN_loss_criterion(pred_fake, False)

        # Real

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        # self.pred_real = self.netD.forward(real_AB)
        self.pred_real = self.dis_model(real_AB)
        pred_real = self.pred_real.cpu()
        self.loss_D_real = self.GAN_loss_criterion(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        return self.loss_D

        # self.loss_D.backward()
        
    def _backward_G(self):
        # First, G(A) should fake the discriminator

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # pred_fake = self.netD.forward(fake_AB)
        pred_fake_G = self.dis_model(fake_AB)
        self.loss_G_GAN = self.GAN_loss_criterion(pred_fake_G.cpu(), True)

        # Second, G(A) = B

        self.loss_G_L1 = self.loss_criterion(self.fake_B, self.real_B) * 300
        
        # Tird, ARNet(R(A))
        self.loss_R_L1 = self.loss_criterion(self.refine_residual.detach(), self.real_residual)*10

        # forth contextual loss

        self.loss_G_con = self.contextual_loss(self.fake_B, self.real_B)

       
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_R_L1 + self.loss_G_con

        return self.loss_G

        # self.loss_G.backward()

    def _forward_pass(self, model, input, target, weight=None):
        # forward passmodel
        #My method (ARNetV3)
        output = model(input)
        loss = self.loss_criterion(output, target)
        # pre_output = self.model(input)
        # residual = target-input
        # pre_res = pre_output-input
        # estimated_res = self.refine_model(pre_res)
        # final_output = estimated_res+input
        # if weight is None:
        #     loss_1 = self.loss_criterion(final_output, target)
        #     loss_2 = self.loss_criterion(estimated_res,residual)
        # else:
        #     loss = self.loss_criterion(pre_output, target, weight)

        # return final_output, pre_output, residual, estimated_res, loss_1, loss_2
        return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        # if isinstance(self.model, nn.DataParallel):
        #     state_dict = self.model.module.state_dict()
        # else:
        #     state_dict = self.model.state_dict()
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
            refine_state_dict = self.refine_model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
            refine_state_dict = self.refine_model.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")
        #save model state 
        utils.save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer_pre.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir)
        #save refine model state
        utils.save_joint_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': refine_state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer_refine.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir,model_name='refine')

    def _log_lr(self):
        lr = self.optimizer_pre.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)
        # lr_refine = self.optimizer_refine.param_groups[0]['lr']
        # self.writer.add_scalar('Refine learning_rate', lr_refine, self.num_iterations)

    def _log_stats(self, phase, loss_avg, res_loss_avg, disc_loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_disc_loss_avg': disc_loss_avg,
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_res_loss_avg': res_loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }  

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            # print(name,value)
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            # self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, pre_output, residual, estimated_res, prefix=''):

        inputs_map = {
            'inputs': input,
            'targets': target,
            'final_output': prediction,
            'pre_output': pre_output,
            'residual':residual,
            'estimated_res':estimated_res
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)


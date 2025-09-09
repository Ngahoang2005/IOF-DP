# import logging
# import numpy as np
# import torch
# import os
# from torch import nn
# from torch.serialization import load
# from tqdm import tqdm
# from torch import optim
# from torch.nn import functional as F
# from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
# from utils.data_manager import DummyDataset
# from utils.inc_net import IncrementalNet, CosineIncrementalNet, Drift_Estimator, ALClassifier
# from models.base import BaseLearner
# from utils.toolkit import target2onehot, tensor2numpy
# from torchvision import datasets, transforms
# from utils.autoaugment import CIFAR10Policy


# init_epoch = 20
# init_lr = 0.1
# init_milestones = [60, 120, 160]
# init_lr_decay = 0.1
# init_weight_decay = 0.0005

# # cifar100
# epochs = 20 
# lrate = 0.05
# milestones = [45, 90]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8
# T = 2
# lamda = 10

# # Tiny-ImageNet200
# # epochs = 100
# # lrate = 0.001
# # milestones = [45, 90]
# # lrate_decay = 0.1
# # batch_size = 128
# # weight_decay = 2e-4
# # num_workers = 8
# # T = 2
# # lamda = 10

# # imagenet100
# # epochs = 100
# # lrate = 0.05
# # milestones = [45, 90]
# # lrate_decay = 0.1
# # batch_size = 128
# # weight_decay = 2e-4
# # num_workers = 8
# # T = 2
# # lamda = 5


# # fine-grained dataset
# # init_lr = 0.01
# # lrate = 0.005
# # lamda = 20

# # refer to supplementary materials for other dataset training settings

# EPSILON = 1e-8
# class IPTScore:
#     def __init__(
#         self, model, 
#         beta1:float, 
#         beta2:float, 
#         tau:float,  # 01mask转换
#         taylor = None, # 表示用几阶梯度来做为重要性指标 param_second, param_first, param_mix
#     ):
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.model = model
#         self.ipt_outer = {} 
#         self.exp_avg_ipt_outer = {}
#         self.exp_avg_unc_outer = {}
#         self.ipt_inner = {} 
#         self.exp_avg_ipt_inner = {}
#         self.exp_avg_unc_inner = {}
#         self.taylor = taylor
#         self.tau = tau
#         print(f"self.taylor is: {self.taylor}")
#         print(f"self.tau is: {self.tau}")
#         assert (self.beta1<1 and self.beta1>0)
#         assert (self.beta2<1 and self.beta2>0)


#     def update_ipt_outer(self, model, global_step): 
#         for n,p in model.named_parameters():
#             if "fc" in n:   # bỏ qua tất cả tham số có 'fc' trong tên
#                 continue
#             if p.requires_grad:
#                 if torch.isnan(p.grad).any():
#                     print(f"{n},外层循环梯度中存在 NaN 值")
#                     #print(p.grad)
#                     print(f"step is {global_step}")
#                     sys.exit(1) 
#                 if n not in self.ipt_outer:
#                     self.ipt_outer[n] = torch.zeros_like(p)
#                     self.exp_avg_ipt_outer[n] = torch.zeros_like(p) 
#                     self.exp_avg_unc_outer[n] = torch.zeros_like(p) 
#                 with torch.no_grad():
#                     # Calculate sensitivity 
#                     self.ipt_outer[n] = (p * p.grad).abs().detach()
#                     if self.taylor in ['param_second']:
#                         self.ipt_outer[n] = (p * p.grad * p * p.grad).abs().detach()
#                     elif self.taylor in ['param_mix']:
#                         self.ipt_outer[n] = (p * p.grad - 0.5 * p * p.grad * p * p.grad).abs().detach()
#                     self.exp_avg_ipt_outer[n] = self.beta1 * self.exp_avg_ipt_outer[n] + (1-self.beta1)*self.ipt_outer[n]
#                     # Update uncertainty 
#                     self.exp_avg_unc_outer[n] = self.beta2 * self.exp_avg_unc_outer[n] + (1-self.beta2)*(self.ipt_outer[n]-self.exp_avg_ipt_outer[n]).abs()
#     def update_ipt_inner(self, model, global_step): 
#         for n,p in model.named_parameters():
#             if "fc" in n:   # bỏ qua tất cả tham số có 'fc' trong tên
#                 continue
#             if p.requires_grad:
#                 if torch.isnan(p.grad).any():
#                     print(f"{n},梯度中存在 NaN 值")
#                     #print(p.grad)
#                     print(f"step is {global_step}")
#                     break 
#                 if n not in self.ipt_inner:
#                     self.ipt_inner[n] = torch.zeros_like(p)
#                     self.exp_avg_ipt_inner[n] = torch.zeros_like(p) 
#                     self.exp_avg_unc_inner[n] = torch.zeros_like(p) 
#                 with torch.no_grad():
#                     # Calculate sensitivity 
#                     self.ipt_inner[n] = (p * p.grad).abs().detach()
#                     if self.taylor in ['param_second']:
#                         self.ipt_inner[n] = (p * p.grad * p * p.grad).abs().detach()
#                     elif self.taylor in ['param_mix']:
#                         self.ipt_inner[n] = (p * p.grad - 0.5 * p * p.grad * p * p.grad).abs().detach()
#                     # Update sensitivity 
#                     self.exp_avg_ipt_inner[n] = self.beta1 * self.exp_avg_ipt_inner[n] + (1-self.beta1)*self.ipt_inner[n]
#                     # Update uncertainty 
#                     self.exp_avg_unc_inner[n] = self.beta2 * self.exp_avg_unc_inner[n] + (1-self.beta2)*(self.ipt_inner[n]-self.exp_avg_ipt_inner[n]).abs()
#     def normalize_importance_scores(self, ipt_score_dic):
    
#         all_scores_tensor = torch.cat([score.flatten() for score in ipt_score_dic.values()])
    
#         min_score = torch.min(all_scores_tensor)
#         max_score = torch.max(all_scores_tensor)
#         normalized_dic = {}
#         for n, score in ipt_score_dic.items():
#             normalized_dic[n] = (score - min_score) / (max_score - min_score)


#         all_scores_tensor = torch.cat([score.flatten() for score in normalized_dic.values()])
#         min_score = torch.min(all_scores_tensor)
#         max_score = torch.max(all_scores_tensor)
#         #sys.exit(1)
#         return normalized_dic

#     def calculate_score_inner(self, p=None, metric="ipt"):
#         assert len(self.exp_avg_ipt_inner) == len(self.exp_avg_unc_inner)
    
#         ipt_score_dic_inner = {}
#         for n in self.exp_avg_ipt_inner:
#             #print(f"name is {n}")
#             #ipt_name_list.append(n)
#             if metric == "ipt":
#                 # Combine the senstivity and uncertainty 
#                 ipt_score = self.exp_avg_ipt_inner[n] * self.exp_avg_unc_inner[n]
#                 #ipt_score = self.exp_avg_ipt_inner[n]
#             elif metric == "mag":
#                 ipt_score = p.abs().detach().clone() 
#             else:
#                 raise ValueError("Unexcptected Metric: %s"%metric)
#             ipt_score_dic_inner[n] = ipt_score
#         assert len(self.exp_avg_ipt_outer) == len(self.exp_avg_unc_outer)
#         ipt_score_dic_outer = {}
#         for n in self.exp_avg_ipt_outer:
#             if metric == "ipt":
#                 ipt_score = self.exp_avg_ipt_outer[n] * self.exp_avg_unc_outer[n]
#                 #ipt_score = self.exp_avg_ipt_outer[n]
#             elif metric == "mag":
#                 ipt_score = p.abs().detach().clone() 
#             else:
#                 raise ValueError("Unexcptected Metric: %s"%metric)
#             ipt_score_dic_outer[n] = ipt_score
#         ipt_score_dic_inner_norm = self.normalize_importance_scores(ipt_score_dic_inner)
#         ipt_score_dic_outer_norm = self.normalize_importance_scores(ipt_score_dic_outer)

#         inner_mask = {}

#         for key in ipt_score_dic_inner_norm:
#             assert key in ipt_score_dic_outer_norm
#             ipt_score_inner = ipt_score_dic_inner_norm[key].cpu().numpy()
#             ipt_score_outer = ipt_score_dic_outer_norm[key].cpu().numpy()

#             exp_term_inner = np.exp(ipt_score_inner / self.tau)
#             exp_term_outer = np.exp(ipt_score_outer / self.tau)
#             denominator = exp_term_inner + exp_term_outer
#             coefficient_inner = exp_term_inner / denominator
#             inner_mask[key] = coefficient_inner
#             inner_mask[key] = torch.tensor(coefficient_inner, device=ipt_score_dic_inner[key].device)
#         return inner_mask

#     def calculate_score_outer(self, p=None, metric="ipt"):
#         assert len(self.exp_avg_ipt_inner) == len(self.exp_avg_unc_inner)

#         ipt_score_dic_inner = {}
#         for n in self.exp_avg_ipt_inner:
#             if metric == "ipt":
#                 # Combine the senstivity and uncertainty 
#                 ipt_score = self.exp_avg_ipt_inner[n] * self.exp_avg_unc_inner[n]
#                 #ipt_score = self.exp_avg_ipt_inner[n]
#             elif metric == "mag":
#                 ipt_score = p.abs().detach().clone() 
#             else:
#                 raise ValueError("Unexcptected Metric: %s"%metric)
        
#             ipt_score_dic_inner[n] = ipt_score

#         assert len(self.exp_avg_ipt_outer) == len(self.exp_avg_unc_outer)
        
        
#         ipt_score_dic_outer = {}
#         for n in self.exp_avg_ipt_outer:
        
#             if metric == "ipt":
            
#                 ipt_score = self.exp_avg_ipt_outer[n] * self.exp_avg_unc_outer[n]
#                 #ipt_score = self.exp_avg_ipt_outer[n]
#             elif metric == "mag":
#                 ipt_score = p.abs().detach().clone() 
#             else:
#                 raise ValueError("Unexcptected Metric: %s"%metric)
          
            
#             ipt_score_dic_outer[n] = ipt_score

     
#         ipt_score_dic_inner_norm = self.normalize_importance_scores(ipt_score_dic_inner)
#         ipt_score_dic_outer_norm = self.normalize_importance_scores(ipt_score_dic_outer)

#         outer_mask = {}

#         for key in ipt_score_dic_inner_norm:
#             assert key in ipt_score_dic_outer_norm
#             ipt_score_inner = ipt_score_dic_inner_norm[key].cpu().numpy()
#             ipt_score_outer = ipt_score_dic_outer_norm[key].cpu().numpy()
         
#             exp_term_inner = np.exp(ipt_score_inner / self.tau)
#             exp_term_outer = np.exp(ipt_score_outer / self.tau)
        
#             denominator = exp_term_inner + exp_term_outer
            
#             coefficient_outer = exp_term_outer / denominator

#             outer_mask[key] = torch.tensor(coefficient_outer, device=ipt_score_dic_outer[key].device)
 

#         return outer_mask
    
#     def update_inner_score(self, model, global_step):

#         self.update_ipt_inner(model, global_step)
    
#     def update_outer_score(self, model, global_step):
      
#         self.update_ipt_outer(model, global_step)

# class LwF(BaseLearner):
#     def __init__(self, args):
#         super().__init__(args)
#         self.args = args
#         if self.args["dataset"] == "imagenet100" or self.args["dataset"] == "imagenet1000":
#             epochs = 100
#             lrate = 0.05
#             milestones = [45, 90]
#             lrate_decay = 0.1
#             batch_size = 128
#             weight_decay = 2e-4
#             num_workers = 8
#             T = 2
#             lamda = 5
#             self.num_per_class = 1300
#         elif self.args["dataset"] == "tinyimagenet200":
#             epochs = 100
#             lrate = 0.001
#             milestones = [45, 90]
#             lrate_decay = 0.1
#             batch_size = 128
#             weight_decay = 2e-4
#             num_workers = 8
#             T = 2
#             lamda = 10
#         print("Number of samples per class:{}".format(self.num_per_class))
#         if self.args["dataset"] == "cub200":
#             init_lr = 0.1
#             lrate = 0.05
#             lamda = 20
#             self.num_per_class = 30
#         if self.args["cosine"]:
#             self._network = CosineIncrementalNet(args, False)
#         else:
#             self._network = IncrementalNet(args, False)

#         self._protos = []
#         self.al_classifier = None
#         if self.args["DPCR"]:
#             self._covs = []
#             self._projectors = []
#         self._old_network = None 
#         self.ipt_score = IPTScore(self._network, beta1=0.55, beta2=0.55, tau=0.1)
#         self.T = args.get("T", 2.0)
#     def after_task(self):
#         self._old_network = self._network.copy().freeze()
#         self._known_classes = self._total_classes
#         if not self.args['resume']:
#             if not os.path.exists(self.args["model_dir"]):
#                 os.makedirs(self.args["model_dir"])
#             self.save_checkpoint("{}".format(self.args["model_dir"]))
        
#     def incremental_train(self, data_manager):
#         self.data_manager = data_manager
#         self._cur_task += 1
#         if self.args['dataset'] == "cifar100":
#             self.data_manager._train_trsf = [
#                 transforms.RandomCrop(32, padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ColorJitter(brightness=63/255),
#                 CIFAR10Policy(),
#                 transforms.ToTensor(),
#             ]
#         elif self.args['dataset'] == "tinyimagenet200":
#             self.data_manager._train_trsf = [
#                 transforms.RandomCrop(64, padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.ToPILImage()
#             ]
#         elif self.args['dataset'] == "imagenet100" or self.args['dataset'] == "cub200":
#             self.data_manager._train_trsf = [
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.ToPILImage()
#             ]
#         self._total_classes = self._known_classes + data_manager.get_task_size(
#             self._cur_task
#         )
#         if self.args["cosine"]:
#             self._network.update_fc(self._total_classes, self._cur_task)
#         else:
#             self._network.update_fc(self._total_classes)

#         if self.al_classifier == None:
#             self.al_classifier = ALClassifier(512, self._total_classes, 0, self._device,args=self.args).to(self._device)
#             for name, param in self.al_classifier.named_parameters():
#                 param.requires_grad = False
#         else:
#             self.al_classifier.augment_class(data_manager.get_task_size(self._cur_task))
#         logging.info(
#             "Learning on {}-{}".format(self._known_classes, self._total_classes)
#         )

#         self.shot = None
#         train_dataset = data_manager.get_dataset(
#             np.arange(self._known_classes, self._total_classes),
#             source="train",
#             mode="train",
#             shot=self.shot
#         )
#         # self.train_dataset = train_dataset
#         self.train_loader = DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
#         )
#         test_dataset = data_manager.get_dataset(
#             np.arange(0, self._total_classes), source="test", mode="test"
#         )
#         self.test_loader = DataLoader(
#             test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
#         )

#         if len(self._multiple_gpus) > 1:
#             self._network = nn.DataParallel(self._network, self._multiple_gpus)
#         self._train(self.train_loader, self.test_loader)
#         if len(self._multiple_gpus) > 1:
#             self._network = self._network.module

#     def _train(self, train_loader, test_loader):
#         resume = self.args['resume']  # set resume=True to use saved checkpoints
#         if self._cur_task == 0:
#             if resume:
#                 print("Loading checkpoint: {}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))
#                 self._network.load_state_dict(torch.load("{}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))["state_dict"], strict=False)
#             self._network.to(self._device)
#             if hasattr(self._network, "module"):
#                 self._network_module_ptr = self._network.module
#             if not resume:
#                 optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=init_lr, weight_decay=init_weight_decay)
#                 scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
#                 self._init_train(train_loader, test_loader, optimizer, scheduler)

#             self._network.eval()
#             pbar = tqdm(enumerate(train_loader), desc='Analytic Learning Phase=' + str(self._cur_task),
#                              total=len(train_loader),
#                              unit='batch')
#             cov = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size).to(self._device)
#             crs_cor = torch.zeros(self.al_classifier.fc.weight.size(1), self._total_classes).to(self._device)
#             with torch.no_grad():
#                 for i, (_, inputs, targets) in pbar:
#                     inputs, targets = inputs.to(self._device), targets.to(self._device)
#                     out_backbone = self._network(inputs)["features"]
#                     out_fe, pred = self.al_classifier(out_backbone)
#                     label_onehot = F.one_hot(targets, self._total_classes).float()
#                     cov += torch.t(out_fe) @ out_fe
#                     crs_cor += torch.t(out_fe) @ (label_onehot)
#             self.al_classifier.cov = self.al_classifier.cov + cov
#             self.al_classifier.R = self.al_classifier.R + cov
#             self.al_classifier.Q = self.al_classifier.Q + crs_cor
#             R_inv = torch.inverse(self.al_classifier.R.cpu()).to(self._device)
#             Delta = R_inv @ self.al_classifier.Q

#             self.al_classifier.fc.weight = torch.nn.parameter.Parameter(
#                     F.normalize(torch.t(Delta.float()), p=2, dim=-1))
#             self._build_protos()
#         else:
#             resume = self.args['resume']
#             if resume:
#                 print("Loading checkpoint: {}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))
#                 self._network.load_state_dict(torch.load("{}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))["state_dict"], strict=False)
#             self._network.to(self._device)
#             if hasattr(self._network, "module"):
#                 self._network_module_ptr = self._network.module
#             if self._old_network is not None:
#                 self._old_network.to(self._device)
#             if not resume:
#                 optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
#                 scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
#                 self._update_representation(train_loader, test_loader, optimizer, scheduler)
#             self._build_protos()                
                    
                    
#             if self.args["DPCR"]:
#                 print('Using DPCR')
#                 self._network.eval()
#                 self.projector = Drift_Estimator(512,False,self.args)
#                 self.projector.to(self._device)
#                 for name, param in self.projector.named_parameters():
#                     param.requires_grad = False
#                 self.projector.eval()
#                 cov_pwdr = self.projector.rg_tssp * torch.eye(self.projector.fe_size).to(self._device)
#                 crs_cor_pwdr = torch.zeros(self.projector.fe_size, self.projector.fe_size).to(self._device)

#                 crs_cor_new = torch.zeros(self.al_classifier.fc.weight.size(1), self._total_classes).to(self._device)
#                 cov_new = torch.zeros(self.projector.fe_size, self.projector.fe_size).to(self._device)
#                 with torch.no_grad():
#                     for i, (_, inputs, targets) in enumerate(train_loader):
#                         inputs, targets = inputs.to(self._device), targets.to(self._device)
#                         feats_old = self._old_network(inputs)["features"]
#                         # print(feats_old)
#                         feats_new = self._network(inputs)["features"]
#                         cov_pwdr += torch.t(feats_old) @ feats_old
#                         cov_new += torch.t(feats_new) @ feats_new
#                         crs_cor_pwdr += torch.t(feats_old) @ (feats_new)
#                         label_onehot = F.one_hot(targets, self._total_classes).float()
#                         crs_cor_new += torch.t(feats_new) @ (label_onehot)
#                 self.projector.cov = cov_pwdr
#                 self.projector.Q = crs_cor_pwdr
#                 R_inv = torch.inverse(cov_pwdr.cpu()).to(self._device)
#                 Delta = R_inv @ crs_cor_pwdr
#                 self.projector.fc.weight = torch.nn.parameter.Parameter(torch.t(Delta.float()))

#                 cov_prime = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size).to(self._device)
#                 Q_prime = torch.zeros(self.al_classifier.fe_size, self.al_classifier.num_classes).to(self._device)

#                 for class_idx in range(0, self._known_classes):
#                     W = self.projector.get_weight() @ self._projectors[class_idx]
#                     cov_idx = self._covs[class_idx]
#                     cov_prime_idx = torch.t(W) @ cov_idx @ W
#                     label = class_idx
#                     label_onehot = F.one_hot(torch.tensor(label).long().to(self._device), self._total_classes).float()
#                     cor_prime_idx = self.num_per_class * (torch.t(W) @ torch.t(
#                         self._protos[class_idx].view(1, self.al_classifier.fe_size))) @ label_onehot.view(1, self._total_classes)
#                     cov_prime += cov_prime_idx
#                     Q_prime += cor_prime_idx
#                     self._covs[class_idx] = cov_prime_idx
#                     self._projectors[class_idx] = self.get_projector_svd(cov_prime_idx)
#                     self._protos[class_idx] = self._protos[class_idx] @ W

#                 R_prime = cov_prime + self.al_classifier.gamma * torch.eye(self.al_classifier.fe_size).to(self._device)
#                 self.al_classifier.cov = cov_prime + cov_new
#                 self.al_classifier.Q = Q_prime + crs_cor_new
#                 self.al_classifier.R = R_prime+ cov_new
#                 R_inv = torch.inverse(self.al_classifier.R.cpu()).to(self._device)
#                 Delta = R_inv @ self.al_classifier.Q
#                 self.al_classifier.fc.weight = torch.nn.parameter.Parameter(
#                         F.normalize(torch.t(Delta.float()), p=2, dim=-1))

#     # SVD for calculating the W_c
#     def get_projector_svd(self, raw_matrix, all_non_zeros=True):
#         V, S, VT = torch.svd(raw_matrix)
#         if all_non_zeros:
#             non_zeros_idx = torch.where(S > 0)[0]
#             left_eign_vectors = V[:, non_zeros_idx]

#         else:
#             left_eign_vectors = V[:, :512]
#         projector = left_eign_vectors @ torch.t(left_eign_vectors)
#         return projector

#     def _build_protos(self):
#         if self.args["DPCR"]:
#             for class_idx in range(self._known_classes, self._total_classes):
#                 data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
#                                                                            source='train',
#                                                                            mode='test', shot=self.shot, ret_data=True)
#                 idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#                 vectors, _ = self._extract_vectors(idx_loader)
#                 class_mean = np.mean(vectors, axis=0)  # vectors.mean(0)
#                 cov = np.dot(np.transpose(vectors),vectors)
#                 self._protos.append(torch.tensor(class_mean).to(self._device))
#                 self._covs.append(torch.tensor(cov).to(self._device))
#                 self._projectors.append(self.get_projector_svd(self._covs[class_idx]))


#     def _init_train(self, train_loader, test_loader, optimizer, scheduler):
#         prog_bar = tqdm(range(init_epoch))
#         for _, epoch in enumerate(prog_bar):
#             self._network.train()
#             losses = 0.0
#             correct, total = 0, 0
#             for i, (_, inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(self._device), targets.to(self._device)
#                 logits = self._network(inputs)["logits"]

#                 loss = F.cross_entropy(logits, targets)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 losses += loss.item()

#                 _, preds = torch.max(logits, dim=1)
#                 correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                 total += len(targets)

#             scheduler.step()
#             train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

#             if epoch % 25 == 0:
#                 test_acc = self._compute_accuracy(self._network, test_loader)
#                 info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
#                     self._cur_task,
#                     epoch + 1,
#                     init_epoch,
#                     losses / len(train_loader),
#                     train_acc,
#                     test_acc,
#                 )
#             else:
#                 info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
#                     self._cur_task,
#                     epoch + 1,
#                     init_epoch,
#                     losses / len(train_loader),
#                     train_acc,
#                 )
#             prog_bar.set_description(info)

#         logging.info(info)


#     def update_parameters_with_task_vectors(self, theta_t, delta_in, delta_out):
#         inner_mask = self.ipt_score.calculate_score_inner(metric="ipt")
#         outer_mask = self.ipt_score.calculate_score_outer(metric="ipt")
        
#         for n in inner_mask:
#             inner = inner_mask[n]
#             outer = outer_mask[n]
#             assert inner.shape == outer.shape, f"Mismatched shape for {n}: {inner.shape} vs {outer.shape}"
#             both_one = (inner == 1) & (outer == 1)
#             inner[both_one] = 0.4
#             outer[both_one] = 0.6
            
#             both_zero = (inner == 0) & (outer == 0)
#             inner[both_zero] = 0.5
#             outer[both_zero] = 0.5
        
#         keys_inner_mask = set(inner_mask.keys())
#         keys_delta_in = set(delta_in.keys())
#         keys_delta_out = set(delta_out.keys())
#         keys_outer_mask = set(outer_mask.keys())
#         keys_theta_t = set(theta_t.keys())
#         assert keys_inner_mask == keys_delta_in == keys_delta_out == keys_outer_mask == keys_theta_t, (
#             f"Key mismatch: inner_mask keys: {keys_inner_mask}, "
#             f"delta_in keys: {keys_delta_in}, "
#             f"delta_out keys: {keys_delta_out}, "
#             f"outer_mask keys: {keys_outer_mask}, "
#             f"theta_t keys: {keys_theta_t}"
#         )
#         final_delta = {n: inner_mask[n] * delta_in[n] + outer_mask[n] * delta_out[n] for n in theta_t}
#         with torch.no_grad():
#             for n, p in self._network.named_parameters():
#                 if n in final_delta:
#                     p.copy_(theta_t[n] + final_delta[n])
    

#     def _update_representation(self, train_loader, test_loader, optimizer, scheduler): 
#         #self.ipt_score.empty_inner_score()
#         #self.ipt_score.empty_outer_score()

#         prog_bar = tqdm(range(epochs))
#         for epoch in prog_bar:
#             self._network.train()
#             losses = 0.0
#             correct, total = 0, 0

#             data_iter = iter(train_loader)

#             for cycle in range(12):  # 32 chu kỳ
#                 # === 4 bước INNER ===
#                 theta_t = {n: p.clone().detach() for n, p in self._network.named_parameters() if "fc" not in n}
#                 for _ in range(5 - int(self._cur_task / 2)):
#                     try:
#                         _, inputs, targets = next(data_iter)
#                     except StopIteration:
#                         data_iter = iter(train_loader)
#                         _, inputs, targets = next(data_iter)


#                     inputs, targets = inputs.to(self._device), targets.to(self._device)
#                     student_outputs = self._network(inputs)["logits"]
#                     fake_targets = targets - self._known_classes
#                     loss_inner = F.cross_entropy(student_outputs[:, self._known_classes:], fake_targets)

#                     optimizer.zero_grad()
#                     loss_inner.backward()
#                     self.ipt_score.update_inner_score(self._network, epoch)
#                     optimizer.step()
    
#                     losses += loss_inner.item()
#                     _, preds = torch.max(student_outputs, dim=1)
#                     correct += preds.eq(targets).cpu().sum().item()
#                     total += targets.size(0)
#                 theta_after_inner = {n: p.clone().detach() for n, p in self._network.named_parameters() if "fc" not in n}
#                 delta_in = {n: theta_after_inner[n] - theta_t[n] for n in theta_t}
#                 # === 1 bước OUTER ===
#                 for _ in range(5): 
#                     inputs, targets = inputs.to(self._device), targets.to(self._device)
#                     logits = self._network(inputs)["logits"]
#                     fake_targets = targets - self._known_classes
#                     loss_clf = F.cross_entropy(
#                     logits[:, self._known_classes :], fake_targets
#                     )
#                     loss_kd = _KD_loss(
#                     logits[:, : self._known_classes],
#                     self._old_network(inputs)["logits"],
#                     T,
#                     )
#                     loss = 10 * loss_kd + loss_clf
#                     optimizer.zero_grad()
#                     loss.backward()
#                     self.ipt_score.update_outer_score(self._network, epoch)
#                     optimizer.step()

#                     losses += loss.item()
#                     with torch.no_grad():
#                         _, preds = torch.max(logits, dim=1)
#                         correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                         total += len(targets)
#                 theta_after_outer = {n: p.clone().detach() for n, p in self._network.named_parameters() if "fc" not in n}
#                 delta_out = {n: theta_after_outer[n] - theta_after_inner[n] for n in theta_t}
#                 self.update_parameters_with_task_vectors(theta_t, delta_in, delta_out) 
#             # ---- epoch end ----
#             scheduler.step()
#             train_acc = np.around(tensor2numpy(torch.tensor(correct)) * 100 / total, decimals=2)
#             test_acc = self._compute_accuracy(self._network, test_loader)
#             info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
#                 self._cur_task,
#                 epoch + 1,
#                 epochs,
#                 losses / len(train_loader),
#                 train_acc,
#                 test_acc,
#             )
#             prog_bar.set_description(info)
#         logging.info(info)

# def _KD_loss(pred, soft, T):
#     # pred = torch.log_softmax(pred / T, dim=1)
#     soft = torch.softmax(soft / T, dim=1)
#     return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

import logging
import numpy as np
import torch
import os
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from utils.data_manager import DummyDataset
from utils.inc_net import IncrementalNet, CosineIncrementalNet, Drift_Estimator, ALClassifier
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy


init_epoch = 20
init_lr = 0.1
init_milestones = [60, 120, 160]
init_lr_decay = 0.1
init_weight_decay = 0.0005

# cifar100
epochs = 20
lrate = 0.05
milestones = [45, 90]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2
lamda = 10

# Tiny-ImageNet200
# epochs = 100
# lrate = 0.001
# milestones = [45, 90]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8
# T = 2
# lamda = 10

# imagenet100
# epochs = 100
# lrate = 0.05
# milestones = [45, 90]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8
# T = 2
# lamda = 5


# fine-grained dataset
# init_lr = 0.01
# lrate = 0.005
# lamda = 20

# refer to supplementary materials for other dataset training settings

EPSILON = 1e-8
class IPTScore:
    def __init__(
        self, model, 
        beta1:float, 
        beta2:float, 
        tau:float,  # 01mask转换
        taylor = None, # 表示用几阶梯度来做为重要性指标 param_second, param_first, param_mix
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.model = model
        self.ipt_outer = {} 
        self.exp_avg_ipt_outer = {}
        self.exp_avg_unc_outer = {}
        self.ipt_inner = {} 
        self.exp_avg_ipt_inner = {}
        self.exp_avg_unc_inner = {}
        self.taylor = taylor
        self.tau = tau
        print(f"self.taylor is: {self.taylor}")
        print(f"self.tau is: {self.tau}")
        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)


    def update_ipt_outer(self, model, global_step): 
        for n,p in model.named_parameters():
            if "fc" in n:   # bỏ qua tất cả tham số có 'fc' trong tên
                continue
            if p.requires_grad:
                if torch.isnan(p.grad).any():
                    print(f"{n},外层循环梯度中存在 NaN 值")
                    #print(p.grad)
                    print(f"step is {global_step}")
                    sys.exit(1) 
                if n not in self.ipt_outer:
                    self.ipt_outer[n] = torch.zeros_like(p)
                    self.exp_avg_ipt_outer[n] = torch.zeros_like(p) 
                    self.exp_avg_unc_outer[n] = torch.zeros_like(p) 
                with torch.no_grad():
                    # Calculate sensitivity 
                    self.ipt_outer[n] = (p * p.grad).abs().detach()
                    if self.taylor in ['param_second']:
                        self.ipt_outer[n] = (p * p.grad * p * p.grad).abs().detach()
                    elif self.taylor in ['param_mix']:
                        self.ipt_outer[n] = (p * p.grad - 0.5 * p * p.grad * p * p.grad).abs().detach()
                    self.exp_avg_ipt_outer[n] = self.beta1 * self.exp_avg_ipt_outer[n] + (1-self.beta1)*self.ipt_outer[n]
                    # Update uncertainty 
                    self.exp_avg_unc_outer[n] = self.beta2 * self.exp_avg_unc_outer[n] + (1-self.beta2)*(self.ipt_outer[n]-self.exp_avg_ipt_outer[n]).abs()
    def update_ipt_inner(self, model, global_step): 
        for n,p in model.named_parameters():
            if "fc" in n:   # bỏ qua tất cả tham số có 'fc' trong tên
                continue
            if p.requires_grad:
                if torch.isnan(p.grad).any():
                    print(f"{n},梯度中存在 NaN 值")
                    #print(p.grad)
                    print(f"step is {global_step}")
                    break 
                if n not in self.ipt_inner:
                    self.ipt_inner[n] = torch.zeros_like(p)
                    self.exp_avg_ipt_inner[n] = torch.zeros_like(p) 
                    self.exp_avg_unc_inner[n] = torch.zeros_like(p) 
                with torch.no_grad():
                    # Calculate sensitivity 
                    self.ipt_inner[n] = (p * p.grad).abs().detach()
                    if self.taylor in ['param_second']:
                        self.ipt_inner[n] = (p * p.grad * p * p.grad).abs().detach()
                    elif self.taylor in ['param_mix']:
                        self.ipt_inner[n] = (p * p.grad - 0.5 * p * p.grad * p * p.grad).abs().detach()
                    # Update sensitivity 
                    self.exp_avg_ipt_inner[n] = self.beta1 * self.exp_avg_ipt_inner[n] + (1-self.beta1)*self.ipt_inner[n]
                    # Update uncertainty 
                    self.exp_avg_unc_inner[n] = self.beta2 * self.exp_avg_unc_inner[n] + (1-self.beta2)*(self.ipt_inner[n]-self.exp_avg_ipt_inner[n]).abs()
    def normalize_importance_scores(self, ipt_score_dic):
    
        all_scores_tensor = torch.cat([score.flatten() for score in ipt_score_dic.values()])
    
        min_score = torch.min(all_scores_tensor)
        max_score = torch.max(all_scores_tensor)
        normalized_dic = {}
        for n, score in ipt_score_dic.items():
            normalized_dic[n] = (score - min_score) / (max_score - min_score)


        all_scores_tensor = torch.cat([score.flatten() for score in normalized_dic.values()])
        min_score = torch.min(all_scores_tensor)
        max_score = torch.max(all_scores_tensor)
        #sys.exit(1)
        return normalized_dic

    def calculate_score_inner(self, p=None, metric="ipt"):
        assert len(self.exp_avg_ipt_inner) == len(self.exp_avg_unc_inner)
    
        ipt_score_dic_inner = {}
        for n in self.exp_avg_ipt_inner:
            #print(f"name is {n}")
            #ipt_name_list.append(n)
            if metric == "ipt":
                # Combine the senstivity and uncertainty 
                ipt_score = self.exp_avg_ipt_inner[n] * self.exp_avg_unc_inner[n]
                #ipt_score = self.exp_avg_ipt_inner[n]
            elif metric == "mag":
                ipt_score = p.abs().detach().clone() 
            else:
                raise ValueError("Unexcptected Metric: %s"%metric)
            ipt_score_dic_inner[n] = ipt_score
        assert len(self.exp_avg_ipt_outer) == len(self.exp_avg_unc_outer)
        ipt_score_dic_outer = {}
        for n in self.exp_avg_ipt_outer:
            if metric == "ipt":
                ipt_score = self.exp_avg_ipt_outer[n] * self.exp_avg_unc_outer[n]
                #ipt_score = self.exp_avg_ipt_outer[n]
            elif metric == "mag":
                ipt_score = p.abs().detach().clone() 
            else:
                raise ValueError("Unexcptected Metric: %s"%metric)
            ipt_score_dic_outer[n] = ipt_score
        ipt_score_dic_inner_norm = self.normalize_importance_scores(ipt_score_dic_inner)
        ipt_score_dic_outer_norm = self.normalize_importance_scores(ipt_score_dic_outer)

        inner_mask = {}

        for key in ipt_score_dic_inner_norm:
            assert key in ipt_score_dic_outer_norm
            ipt_score_inner = ipt_score_dic_inner_norm[key].cpu().numpy()
            ipt_score_outer = ipt_score_dic_outer_norm[key].cpu().numpy()

            exp_term_inner = np.exp(ipt_score_inner / self.tau)
            exp_term_outer = np.exp(ipt_score_outer / self.tau)
            denominator = exp_term_inner + exp_term_outer
            coefficient_inner = exp_term_inner / denominator
            inner_mask[key] = coefficient_inner
            inner_mask[key] = torch.tensor(coefficient_inner, device=ipt_score_dic_inner[key].device)
        return inner_mask

    def calculate_score_outer(self, p=None, metric="ipt"):
        assert len(self.exp_avg_ipt_inner) == len(self.exp_avg_unc_inner)

        ipt_score_dic_inner = {}
        for n in self.exp_avg_ipt_inner:
            if metric == "ipt":
                # Combine the senstivity and uncertainty 
                ipt_score = self.exp_avg_ipt_inner[n] * self.exp_avg_unc_inner[n]
                #ipt_score = self.exp_avg_ipt_inner[n]
            elif metric == "mag":
                ipt_score = p.abs().detach().clone() 
            else:
                raise ValueError("Unexcptected Metric: %s"%metric)
        
            ipt_score_dic_inner[n] = ipt_score

        assert len(self.exp_avg_ipt_outer) == len(self.exp_avg_unc_outer)
        
        
        ipt_score_dic_outer = {}
        for n in self.exp_avg_ipt_outer:
        
            if metric == "ipt":
            
                ipt_score = self.exp_avg_ipt_outer[n] * self.exp_avg_unc_outer[n]
                #ipt_score = self.exp_avg_ipt_outer[n]
            elif metric == "mag":
                ipt_score = p.abs().detach().clone() 
            else:
                raise ValueError("Unexcptected Metric: %s"%metric)
          
            
            ipt_score_dic_outer[n] = ipt_score

     
        ipt_score_dic_inner_norm = self.normalize_importance_scores(ipt_score_dic_inner)
        ipt_score_dic_outer_norm = self.normalize_importance_scores(ipt_score_dic_outer)

        outer_mask = {}

        for key in ipt_score_dic_inner_norm:
            assert key in ipt_score_dic_outer_norm
            ipt_score_inner = ipt_score_dic_inner_norm[key].cpu().numpy()
            ipt_score_outer = ipt_score_dic_outer_norm[key].cpu().numpy()
         
            exp_term_inner = np.exp(ipt_score_inner / self.tau)
            exp_term_outer = np.exp(ipt_score_outer / self.tau)
        
            denominator = exp_term_inner + exp_term_outer
            
            coefficient_outer = exp_term_outer / denominator

            outer_mask[key] = torch.tensor(coefficient_outer, device=ipt_score_dic_outer[key].device)
 

        return outer_mask
    
    def update_inner_score(self, model, global_step):

        self.update_ipt_inner(model, global_step)
    
    def update_outer_score(self, model, global_step):
      
        self.update_ipt_outer(model, global_step)

class LwF(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if self.args["dataset"] == "imagenet100" or self.args["dataset"] == "imagenet1000":
            epochs = 100
            lrate = 0.05
            milestones = [45, 90]
            lrate_decay = 0.1
            batch_size = 128
            weight_decay = 2e-4
            num_workers = 8
            T = 2
            lamda = 5
            self.num_per_class = 1300
        elif self.args["dataset"] == "tinyimagenet200":
            epochs = 100
            lrate = 0.001
            milestones = [45, 90]
            lrate_decay = 0.1
            batch_size = 128
            weight_decay = 2e-4
            num_workers = 8
            T = 2
            lamda = 10
        print("Number of samples per class:{}".format(self.num_per_class))
        if self.args["dataset"] == "cub200":
            init_lr = 0.1
            lrate = 0.05
            lamda = 20
            self.num_per_class = 30
        if self.args["cosine"]:
            self._network = CosineIncrementalNet(args, False)
        else:
            self._network = IncrementalNet(args, False)

        self._old_network = None 
        self.ipt_score = IPTScore(self._network, beta1=0.55, beta2=0.55, tau=0.1)
        self.T = args.get("T", 2.0)
    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        if not self.args['resume']:
            if not os.path.exists(self.args["model_dir"]):
                os.makedirs(self.args["model_dir"])
            self.save_checkpoint("{}".format(self.args["model_dir"]))
        
    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self.args['dataset'] == "cifar100":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ]
        elif self.args['dataset'] == "tinyimagenet200":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        elif self.args['dataset'] == "imagenet100" or self.args['dataset'] == "cub200":
            self.data_manager._train_trsf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        if self.args["cosine"]:
            self._network.update_fc(self._total_classes, self._cur_task)
        else:
            self._network.update_fc(self._total_classes)

        # if self.al_classifier == None:
        #     self.al_classifier = ALClassifier(512, self._total_classes, 0, self._device,args=self.args).to(self._device)
        #     for name, param in self.al_classifier.named_parameters():
        #         param.requires_grad = False
        # else:
        #     self.al_classifier.augment_class(data_manager.get_task_size(self._cur_task))
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        #self.shot = None
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train"
            #shot=self.shot
        )
        # self.train_dataset = train_dataset
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)


    def update_parameters_with_task_vectors(self, theta_t, delta_in, delta_out):
        inner_mask = self.ipt_score.calculate_score_inner(metric="ipt")
        outer_mask = self.ipt_score.calculate_score_outer(metric="ipt")
        
        for n in inner_mask:
            inner = inner_mask[n]
            outer = outer_mask[n]
            assert inner.shape == outer.shape, f"Mismatched shape for {n}: {inner.shape} vs {outer.shape}"
            both_one = (inner == 1) & (outer == 1)
            inner[both_one] = 0.4
            outer[both_one] = 0.6
            
            both_zero = (inner == 0) & (outer == 0)
            inner[both_zero] = 0.5
            outer[both_zero] = 0.5
        
        keys_inner_mask = set(inner_mask.keys())
        keys_delta_in = set(delta_in.keys())
        keys_delta_out = set(delta_out.keys())
        keys_outer_mask = set(outer_mask.keys())
        keys_theta_t = set(theta_t.keys())
        assert keys_inner_mask == keys_delta_in == keys_delta_out == keys_outer_mask == keys_theta_t, (
            f"Key mismatch: inner_mask keys: {keys_inner_mask}, "
            f"delta_in keys: {keys_delta_in}, "
            f"delta_out keys: {keys_delta_out}, "
            f"outer_mask keys: {keys_outer_mask}, "
            f"theta_t keys: {keys_theta_t}"
        )
        final_delta = {n: inner_mask[n] * delta_in[n] + outer_mask[n] * delta_out[n] for n in theta_t}
        with torch.no_grad():
            for n, p in self._network.named_parameters():
                if n in final_delta:
                    p.copy_(theta_t[n] + final_delta[n])
    

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler): 
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            data_iter = iter(train_loader)

            for cycle in range(12):  # 32 chu kỳ
                # === 4 bước INNER ===
                theta_t = {n: p.clone().detach() for n, p in self._network.named_parameters() if "fc" not in n}
                for _ in range(5 - int(self._cur_task / 2)):
                    try:
                        _, inputs, targets = next(data_iter)
                    except StopIteration:
                        data_iter = iter(train_loader)
                        _, inputs, targets = next(data_iter)


                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    student_outputs = self._network(inputs)["logits"]
                    fake_targets = targets - self._known_classes
                    loss_inner = F.cross_entropy(student_outputs[:, self._known_classes:], fake_targets)

                    optimizer.zero_grad()
                    loss_inner.backward()
                    self.ipt_score.update_inner_score(self._network, epoch)
                    optimizer.step()
    
                    losses += loss_inner.item()
                    _, preds = torch.max(student_outputs, dim=1)
                    correct += preds.eq(targets).cpu().sum().item()
                    total += targets.size(0)
                theta_after_inner = {n: p.clone().detach() for n, p in self._network.named_parameters() if "fc" not in n}
                delta_in = {n: theta_after_inner[n] - theta_t[n] for n in theta_t}
                # === 1 bước OUTER ===
                for _ in range(5): 
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    logits = self._network(inputs)["logits"]
                    fake_targets = targets - self._known_classes
                    loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                    )
                    loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                    )
                    loss = 10 * loss_kd + loss_clf
                    optimizer.zero_grad()
                    loss.backward()
                    self.ipt_score.update_outer_score(self._network, epoch)
                    optimizer.step()

                    losses += loss.item()
                    with torch.no_grad():
                        _, preds = torch.max(logits, dim=1)
                        correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                        total += len(targets)
                theta_after_outer = {n: p.clone().detach() for n, p in self._network.named_parameters() if "fc" not in n}
                delta_out = {n: theta_after_outer[n] - theta_after_inner[n] for n in theta_t}
                self.update_parameters_with_task_vectors(theta_t, delta_in, delta_out) 
            # ---- epoch end ----
            scheduler.step()
            train_acc = np.around(tensor2numpy(torch.tensor(correct)) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
        logging.info(info)

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision #J
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='dino_vits16', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup) 
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.32, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.32),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # losses parameters
    parser.add_argument('--alpha', type=float, default=1.0, help="""alpha parameter defined to 
        weight between sce loss.""")
    parser.add_argument('--beta', type=float, default=1.0, help="""beta parameter defined to 
        weight between sce loss.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=50, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

  # Proportions
    parser.add_argument('--num_classes_proportions', type=int, default=10, 
        help='Número de clases para las proporciones')
    parser.add_argument('--mode', type=str, default='combined', 
        help='Modo de calcular las proporciones')
    parser.add_argument('--proportion_temp', type=float, default=0.1,
        help='Temperatura para las proporciones')
    parser.add_argument('--proportion_weight', type=float, default=0.1,
        help='Peso para la pérdida de proporciones')
    return parser

def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )

    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing proportion loss ... ============
    proportion_loss = ProportionLoss(
        mode='ce',
        alpha=args.alpha, 
        beta=args.beta, 
        epsilon=1e-8
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.amp.GradScaler()

    # Inicializar el reductor de dimension
    proportion_calculator = ProportionCalculator(
        input_dim=args.out_dim,                   
        output_dim=args.num_classes_proportions, 
        mode='soft',                    
        temperature=0.1,                
        # dropout=0.1                    
    ).cuda()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, proportion_loss, proportion_calculator, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

# Calcular proporciones por lote y dataset
def calculate_class_proportions_in_batch(labels, dataset):
    labels_tensor = labels.clone().detach().cuda() 
    class_counts = torch.bincount(labels_tensor, minlength=len(dataset.classes))
    total_samples_in_batch = len(labels_tensor)
    # Normalizar las proporciones dividiendo por el tamaño total del lote
    class_proportions = (class_counts.float() / total_samples_in_batch).half()
    return class_proportions

def calculate_class_proportions_in_dataset(dataset):
    all_labels = torch.tensor(dataset.targets, dtype=torch.long, device='cuda')
    class_counts = torch.bincount(all_labels, minlength=len(dataset.classes))
    # Calcular proporciones globales basadas en el total de imágenes
    total_samples = len(all_labels)
    class_proportions = (class_counts.float() / total_samples).half()
    return class_proportions

# === Mejora en el cálculo de proporciones ===
class ProportionCalculator(nn.Module):
    def __init__(
        self,
        input_dim=65536,
        output_dim=10,
        mode='soft',
        temperature=1.0,
        # dropout=0.1
    ):
        super().__init__()
        self.mode = mode
        self.temperature = temperature
        
        # Creamos el projector como parte del módulo
        if mode == 'soft':
            self.projector = nn.Linear(input_dim, output_dim, bias=False)
        elif mode == 'hard':
            self.projector = nn.Linear(input_dim, output_dim, bias=False)
        else:
            raise ValueError("Mode must be 'soft' or 'hard'")
        
        # Movemos el projector a GPU si está disponible
        if torch.cuda.is_available():
            self.projector = self.projector.cuda()
            
        # Inicialización de pesos (opcional)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(
            self.projector.weight,
            mode='fan_out',
            nonlinearity='relu'
        )
        # if self.projector.bias is not None:
            # nn.init.zeros_(self.projector.bias)
    
    def forward(self, outputs):
        """
        Forward pass con gradiente garantizado
        Args:
            outputs: Tensor de entrada de forma (batch_size, input_dim)
        Returns:
            batch_proportions: Tensor de proporciones promedio por lote
            probs: Probabilidades por muestra (útil para pérdida)
        """
        # Proyección (mantiene el gradiente)
        projected = self.projector(outputs)
        
        # Escalado por temperatura
        scaled = projected / self.temperature
        
        if self.mode == 'soft':
            # Softmax mantiene el gradiente automáticamente
            probs = F.softmax(scaled, dim=1)
            batch_proportions = torch.mean(probs, dim=0)
            
        else:  # mode == 'hard'
            # Gumbel-softmax con straight-through estimator
            probs = F.gumbel_softmax(scaled, tau=1.0, hard=True)
            
            # Calcular proporciones manteniendo el gradiente
            with torch.no_grad():
                # Calculamos índices duros pero no los usamos para el gradiente
                hard_indices = probs.max(dim=1)[1]
                class_counts = torch.bincount(
                    hard_indices,
                    minlength=projected.size(1)
                ).float()
                
                # Normalizamos los conteos
                batch_proportions = class_counts / outputs.size(0)
            
            # Usamos straight-through estimator para el gradiente
            batch_proportions = (
                batch_proportions + probs.mean(dim=0) - probs.mean(dim=0).detach()
            )
        
        return batch_proportions, probs

class SimpleProjector(nn.Module):
    def __init__(self, input_dim=65450, output_dim=10):
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.projector(x)

class ProportionLoss(nn.Module):
    def __init__(self, mode='sce', alpha=1.0, beta=1.0, epsilon=1e-8):
        """
        Args:
            mode: Modo de pérdida ('ce' para Cross Entropy, 'rce' para Reverse Cross Entropy, 'sce' para Symmetric Cross Entropy).
            alpha: Factor de peso CE en modo 'sce'.
            beta: Factor de peso RCE en modo 'sce'.
            epsilon: Valor pequeño para evitar log(0).
        """
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        """
        Calcular la pérdida de acuerdo al modo seleccionado.
        Args:
            inputs: Predicciones del modelo [batch_size, num_classes_proportions].
            targets: Etiquetas verdaderas [batch_size, num_classes_proportions].
        """
        if self.mode == 'ce':
            return self.ce_loss(inputs, targets)
        elif self.mode == 'rce':
            return self.rce_loss(inputs, targets)
        elif self.mode == 'sce':
            return self.alpha * self.ce_loss(inputs, targets) + self.beta * self.rce_loss(inputs, targets)
        else:
            raise ValueError("El modo debe ser 'ce', 'rce' o 'sce'.")

    def ce_loss(self, inputs, targets):
        """
        Calcula la Cross Entropy: H(q, p).
        """
        # Asegurarse de que `inputs` sean probabilidades (p)
        # inputs = F.softmax(inputs, dim=-1)
        inputs = torch.clamp(inputs, self.epsilon, 1 - self.epsilon)
        # Calcular la pérdida de Cross Entropy
        ce_loss = -torch.sum(targets * torch.log(inputs), dim=-1)
        return ce_loss.mean()

    def rce_loss(self, inputs, targets):
        """
        Calcula la Reverse Cross Entropy: H(p, q).
        """
        # Asegurarse de que `targets` estén normalizados como probabilidades (q)
        targets = F.normalize(targets, p=1, dim=-1)
        targets = torch.clamp(targets, self.epsilon, 1 - self.epsilon)
        # Calcular la pérdida de Reverse Cross Entropy
        rce_loss = -torch.sum(inputs * torch.log(targets), dim=-1)
        return rce_loss.mean()

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, proportion_loss, proportion_calculator, args): 
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
                                              
    for it, (images, labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16_scaler is not None):
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss_dino = dino_loss(student_output, teacher_output, epoch)

            # Usar la salida del estudiante para las proporciones
            # Tomamos solo las primeras dos vistas (globales) del estudiante
            student_global_views = student_output[:2 * args.batch_size_per_gpu]

            estimated_proportions_s, probs = proportion_calculator(student_global_views)
            estimated_proportions_t, probs = proportion_calculator(teacher_output)

            '''
            # Calcular proporciones estimadas usando la nueva función
            estimated_proportions_s = calculate_proportions(
                student_output,
                input_dim=args.out_dim,
                output_dim=args.num_classes_proportions, 
                mode='soft', 
                temperature=0.1,  
            )

          # Calcular proporciones estimadas usando la nueva función
            estimated_proportions_t = calculate_proportions(
                teacher_output,
                input_dim=args.out_dim,
                hidden_dim=1024,
                output_dim=args.num_classes_proportions, 
                mode='soft', 
                temperature=args.teacher_temp,
                dropout=0.1, 
            )
            '''

            # Calcular proporciones verdaderas del batch
            true_proportions = calculate_class_proportions_in_batch(labels, data_loader.dataset)
            
            # Calcular pérdida LLP
            loss_llp = proportion_loss(estimated_proportions_s, true_proportions)
            
            # Combinar pérdidas
            loss = loss_dino + loss_llp

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # imprimir información de las salidas (solo una vez)
        if it == 0 and utils.is_main_process():
            print("teacher output shape:", teacher_output.shape)
            print("teacher output type:", teacher_output.dtype)
            print("student output shape:", student_output.shape)
            print("student output type:", student_output.dtype)
        
        # Añadir verificación de distribuciones
        if it % 100 == 0:  # Verificar cada 100 iteraciones
            print("\nDistribuciones en iteración", it)
            print("True proportions in batch:", [f"{x:.4f}" for x in true_proportions.cpu().numpy()])
            print("Estimated proportions (s):", [f"{x:.4f}" for x in estimated_proportions_s.detach().cpu().numpy()])
            print("Estimated proportions (t):", [f"{x:.4f}" for x in estimated_proportions_t.detach().cpu().numpy()])
            print("Loss DINO:", loss_dino.item())
            print("Loss LLP:", loss_llp.item())
            print("Sum true:", true_proportions.sum().item())
            print("Sum estimated (s):", estimated_proportions_s.sum().item())
            print("Sum estimated (t):", estimated_proportions_t.sum().item())

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_dino=loss_dino.item())
        metric_logger.update(loss_llp=loss_llp.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
                   
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.printed_info = False  # Flag to ensure printing only once

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        # impresiones de la salidas función de pérdida
        if not self.printed_info:
            self.printed_info = True
            # Print information about teacher_out and student_out recrops
            for i, crop in enumerate(teacher_out):
                print(f"teacher output función de pérdida crop {i + 1} shape: {crop.shape}")
                print(f"teacher output función de pérdida crop {i + 1} type: {crop.dtype}")
            for i, crop in enumerate(student_out):
                print(f"student output función de pérdida crop {i + 1} shape: {crop.shape}")
                print(f"student output función de pérdida crop {i + 1} type: {crop.dtype}")

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3444, 0.3804, 0.4079), (0.2028, 0.1370, 0.1156)),
        ]) # Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) 

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)

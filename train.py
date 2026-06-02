import logging
import math
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from modules.optimizers import build_optimizer
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


def get_lr(step, base_lr, warmup_steps, decay_steps, min_ratio=0.1):
    """前 warmup_steps 线性升到 base_lr；之后 cosine 衰减到 base_lr*min_ratio，
    在 warmup_steps→decay_steps 区间内完成衰减，超过 decay_steps 恒定在地板值。
    全程仅依赖 global_step，续训按 step 自动接上，无额外状态。"""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= decay_steps:
        return base_lr * min_ratio
    progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def ensure_base_models(model_dir, speech_encoder):
    """训练前确保主模型底模就位：model_dir 内无 G_*/D_* 时，按编码器从 pretrain/ 复制，
    本地缺失且 meta.base_model_dict 配置了直链则下载；都没有则跳过(从零训练)。"""
    import glob
    import shutil
    import urllib.request
    from pretrain.meta import base_model_dict

    if glob.glob(os.path.join(model_dir, "G_*.pth")) or glob.glob(os.path.join(model_dir, "D_*.pth")):
        return  # 已有 checkpoint，续训，不动底模

    os.makedirs(model_dir, exist_ok=True)
    urls = base_model_dict().get(speech_encoder, {})
    for fname in ("G_0.pth", "D_0.pth"):
        dst = os.path.join(model_dir, fname)
        # 本地查找：编码器专属子目录。pretrain 根目录的 G_0/D_0 是 vec768l12 专属(768维)，
        # 不能跨 ssl_dim 复用——否则 pre/content_merge 等不匹配层会被 load_checkpoint 静默
        # 随机初始化，却保留了预训练 decoder，前后端分布错位导致输出全是噪音。
        candidates = [os.path.join("pretrain", speech_encoder, fname)]
        if speech_encoder == "vec768l12":
            candidates.append(os.path.join("pretrain", fname))
        local = next((p for p in candidates if os.path.exists(p)), None)
        if local:
            shutil.copy2(local, dst)
            print(f"[底模] 已复制 {local} -> {dst}")
            continue
        url = urls.get(fname)
        if not url:
            print(f"[底模] 未找到 {fname}：pretrain/ 下无对应文件且未配置下载直链，将从零训练。")
            continue
        try:
            print(f"[底模] 下载 {url} -> {dst}")
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            if os.path.exists(dst):
                os.remove(dst)
            print(f"[底模] 下载 {fname} 失败({e})，将从零训练。")


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)

    # for pytorch on win, backend use gloo    
    dist.init_process_group(backend=  'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem   # If you have enough memory, turn on this option to avoid disk IO and speed up training.
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)
    num_workers = getattr(hps.train, "num_workers", 2)
    if all_in_mem:
        num_workers = 0
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=False,
                              batch_size=hps.train.batch_size, collate_fn=collate_fn,
                              persistent_workers=num_workers > 0)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps, all_in_mem=all_in_mem,vol_aug = False)
        eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
                                 batch_size=1, pin_memory=False,
                                 drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)
    net_d = build_discriminator(hps.model, hps.data).cuda(rank)
    optim_g = build_optimizer(net_g.parameters(), hps.train)
    optim_d = build_optimizer(net_d.parameters(), hps.train)
    # Windows 上 torch>=2.4 的 DDP+gloo backward 会触发 access violation 崩溃，
    # 单卡训练本就不需要 DDP，直接跳过包装。多卡仍走 DDP。
    if n_gpus > 1:
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])

    skip_optimizer = False
    if rank == 0:
        ensure_base_models(hps.model_dir, hps.model.speech_encoder)
    if n_gpus > 1:
        dist.barrier()
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g, skip_optimizer)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        name=utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        global_step=int(name[name.rfind("_")+1:name.rfind(".")])+1
        #global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    if rank == 0:
        # purge_step 让 TensorBoard 隐藏所有 step >= global_step 的旧 event，
        # 避免续训时权重 step 落后于已记录 step 导致曲线回溯。
        writer = SummaryWriter(log_dir=hps.model_dir, purge_step=global_step)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"), purge_step=global_step)

    scaler = GradScaler('cuda', enabled=hps.train.fp16_run and hps.train.half_type != "bf16")

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], scaler,
                               [train_loader, None], None, None)


def build_discriminator(hps_model, hps_data):
    kinds = ["mpd"]
    if getattr(hps_model, "use_cqt_disc", False):
        kinds.append("cqt")
    if getattr(hps_model, "use_mrd_disc", False):
        kinds.append("mrd")
    if getattr(hps_model, "use_mbd_disc", False):
        kinds.append("mbd")

    if kinds == ["mpd"]:
        return MultiPeriodDiscriminator(hps_model.use_spectral_norm)

    from modules.discriminators_cqt import (
        CombinedDiscriminator,
        MultiBandDiscriminator,
        MultiResolutionDiscriminator,
        MultiScaleSubbandCQTDiscriminator,
    )
    discs = [MultiPeriodDiscriminator(hps_model.use_spectral_norm)]
    if "cqt" in kinds:
        discs.append(MultiScaleSubbandCQTDiscriminator({"sampling_rate": hps_data.sampling_rate}))
    if "mrd" in kinds:
        discs.append(MultiResolutionDiscriminator({}))
    if "mbd" in kinds:
        discs.append(MultiBandDiscriminator({}))
    return CombinedDiscriminator(discs)


def train_and_evaluate(rank, epoch, hps, nets, optims, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    half_type = torch.bfloat16 if hps.train.half_type=="bf16" else torch.float16

    base_lr = hps.train.learning_rate
    warmup_steps = hps.train.warmup_epochs * len(train_loader)
    decay_steps = getattr(hps.train, "lr_decay_steps", 100000)
    c_speaker_adv = getattr(hps.train, "c_speaker_adv", 0.0)

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, items in enumerate(train_loader):
        cur_lr = get_lr(global_step, base_lr, warmup_steps, decay_steps)
        for pg in optim_g.param_groups:
            pg['lr'] = cur_lr
        for pg in optim_d.param_groups:
            pg['lr'] = cur_lr
        c, f0, spec, y, spk, lengths, uv,volume = items
        g = spk.cuda(rank, non_blocking=True)
        spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
        c = c.cuda(rank, non_blocking=True)
        f0 = f0.cuda(rank, non_blocking=True)
        uv = uv.cuda(rank, non_blocking=True)
        lengths = lengths.cuda(rank, non_blocking=True)
        volume = volume.cuda(rank, non_blocking=True) if volume is not None else None
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)
        
        with autocast('cuda', enabled=hps.train.fp16_run, dtype=half_type):
            y_hat, ids_slice, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0, speaker_adv_logits = net_g(c, f0, uv, spec, g=g, c_lengths=lengths,
                                                                                                    spec_lengths=lengths,vol = volume)

            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

            with autocast('cuda', enabled=False, dtype=half_type):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
        

        with autocast('cuda', enabled=hps.train.fp16_run, dtype=half_type):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast('cuda', enabled=False, dtype=half_type):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_lf0 = F.mse_loss(pred_lf0, lf0) if getattr(net_g, "module", net_g).use_automatic_f0_prediction else 0
                loss_speaker_adv = F.cross_entropy(speaker_adv_logits, g.squeeze(1)) * c_speaker_adv if speaker_adv_logits is not None else 0
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0 + loss_speaker_adv
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                reference_loss=0
                for i in losses:
                    reference_loss += i
                mel_raw = loss_mel / hps.train.c_mel
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}, mel_raw: {mel_raw.item():.4f}")

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/mel_raw": mel_raw,
                                    "loss/g/kl": loss_kl, "loss/g/lf0": loss_lf0, "loss/g/speaker_adv": loss_speaker_adv,
                                    "loss/g/reference": reference_loss})

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
                }

                if getattr(net_g, "module", net_g).use_automatic_f0_prediction:
                    image_dict.update({
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                              pred_lf0[0, 0, :].detach().cpu().numpy()),
                        "all/norm_lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                                   norm_lf0[0, 0, :].detach().cpu().numpy())
                    })

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)

        global_step += 1

    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv,volume = items
            g = spk[:1].cuda(0)
            spec, y = spec[:1].cuda(0), y[:1].cuda(0)
            c = c[:1].cuda(0)
            f0 = f0[:1].cuda(0)
            uv= uv[:1].cuda(0)
            if volume is not None:
                volume = volume[:1].cuda(0)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            y_hat,_ = getattr(generator, "module", generator).infer(c, f0, uv, g=g,vol = volume)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat[0],
                f"gt/audio_{batch_idx}": y[0]
            })
        image_dict.update({
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()

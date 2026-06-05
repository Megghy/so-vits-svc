import logging
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from eval_metrics import (
    DEFAULT_SPEAKER_SIMILARITY_MODEL,
    SpeakerSimilarityMetric,
    collect_speaker_similarity_scalars,
)
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
from modules.bigvgan_strategy import BigVGANStrategy, is_bigvgan_vocoder

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()
_speaker_similarity_metric = None
_speaker_similarity_metric_key = None

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad_(requires_grad)


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


def build_bigvgan_phase2_optimizer(net_g, hps, vocoder_lr):
    net_g_unwrapped = net_g.module if hasattr(net_g, 'module') else net_g
    if not hasattr(net_g_unwrapped, 'dec') or not hasattr(net_g_unwrapped.dec, 'vocoder'):
        return build_optimizer(net_g.parameters(), hps.train)

    if hasattr(net_g_unwrapped.dec, 'unfreeze_vocoder'):
        net_g_unwrapped.dec.unfreeze_vocoder()
    vocoder_params = list(net_g_unwrapped.dec.vocoder.parameters())
    vocoder_param_ids = {id(p) for p in vocoder_params}
    main_params = [p for p in net_g.parameters() if id(p) not in vocoder_param_ids]
    return build_optimizer([main_params, vocoder_params], hps.train, vocoder_lr=vocoder_lr)


def build_generator_optimizer(net_g, hps, bv_strategy):
    if bv_strategy and bv_strategy.needs_vocoder_finetune():
        return build_bigvgan_phase2_optimizer(net_g, hps, bv_strategy.phase2_vocoder_lr)
    return build_optimizer(net_g.parameters(), hps.train)


def get_grad_clip_norm(hps):
    value = getattr(hps.train, "grad_clip_norm", None)
    return None if value is None else float(value)


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
    pin_memory = torch.cuda.is_available()
    train_loader_kwargs = {
        "num_workers": num_workers,
        "shuffle": False,
        "pin_memory": pin_memory,
        "batch_size": hps.train.batch_size,
        "collate_fn": collate_fn,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = getattr(hps.train, "prefetch_factor", 2)
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps, all_in_mem=all_in_mem,vol_aug = False)
        eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
                                 batch_size=1, pin_memory=pin_memory,
                                 drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)
    net_d = build_discriminator(hps.model, hps.data).cuda(rank)

    # BigVGAN 分阶段策略只作用于 BigVGAN 系声码器；CQT/MRD/MBD 判别器本身是通用训练组件。
    bv_strategy = BigVGANStrategy(hps, hps.model_dir) if is_bigvgan_vocoder(hps) else None
    if rank == 0 and bv_strategy is None and getattr(hps.train, "bigvgan_strategy", None) is not None:
        logger.warning("当前声码器不是 BigVGAN，已忽略 train.bigvgan_strategy；判别器按 train.disc_start_step 启动。")

    optim_g = build_generator_optimizer(net_g, hps, bv_strategy)
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
            # 传入 optims 列表的引用,允许 Phase2 切换时更新
            optims_ref = {"g": optim_g, "d": optim_d}
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], optims_ref, scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval], bv_strategy)
            # Phase2 切换后可能重建了 optim_g,同步回来
            optim_g = optims_ref["g"]
        else:
            optims_ref = {"g": optim_g, "d": optim_d}
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], optims_ref, scaler,
                               [train_loader, None], None, None, bv_strategy)
            optim_g = optims_ref["g"]


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
    return CombinedDiscriminator(discs, kinds=kinds)


def get_vocoder_for_stats(net_g):
    dec = getattr(getattr(net_g, "module", net_g), "dec", None)
    return getattr(dec, "vocoder", dec)


def get_speaker_similarity_metric(hps):
    global _speaker_similarity_metric, _speaker_similarity_metric_key
    model_path = getattr(hps.train, "eval_speaker_model", DEFAULT_SPEAKER_SIMILARITY_MODEL)
    device = "cpu"
    key = (model_path, device)
    if _speaker_similarity_metric is None or _speaker_similarity_metric_key != key:
        _speaker_similarity_metric = SpeakerSimilarityMetric(model_path=model_path, device=device)
        _speaker_similarity_metric_key = key
    return _speaker_similarity_metric


def train_and_evaluate(rank, epoch, hps, nets, optims_ref, scaler, loaders, logger, writers, bv_strategy=None):
    net_g, net_d = nets
    optim_g, optim_d = optims_ref["g"], optims_ref["d"]
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    half_type = torch.bfloat16 if hps.train.half_type=="bf16" else torch.float16

    base_lr = hps.train.learning_rate
    accumulation_steps = max(1, int(getattr(hps.train, "gradient_accumulation_steps", 1)))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    warmup_steps = hps.train.warmup_epochs * updates_per_epoch
    decay_steps = getattr(hps.train, "lr_decay_steps", 100000)
    c_speaker_adv = getattr(hps.train, "c_speaker_adv", 0.0)
    disc_start_step = bv_strategy.get_disc_start_step() if bv_strategy else getattr(hps.train, "disc_start_step", 10000)
    c_bigvgan_mel = getattr(hps.train, "c_bigvgan_mel", 45.0)
    is_combined_disc = hasattr(net_d, "kinds")

    grad_clip_norm = get_grad_clip_norm(hps)
    use_mel_loss = bv_strategy.use_mel_loss if bv_strategy else True
    use_bigvgan_mel_loss = bv_strategy.use_bigvgan_mel_loss if bv_strategy else True

    recent_mel_losses = []

    # 默认 MPD 从训练开始启用；disc_start_step 只控制 CQT/MRD/MBD 这类附加判别器。
    # no-GAN warmup(前 N 步彻底关闭判别器)仅 BigVGAN 策略提供，非 BigVGAN 不开。
    disc_warmup_steps = bv_strategy.disc_warmup_steps if bv_strategy else 0

    # BigVGAN 两段式监督用的 mel:必须用官方权重的 mel 定义(128band/n_fft2048/hop512/
    # win2048/sr44100/fmin0/fmax=None),与 hps.data 的 80-band 配置无关,否则又制造分布错位。
    bigvgan_mel_spectrogram = None
    if getattr(hps.model, "vocoder_name", "") in ("bigvgan", "bigvgan-v2", "nsf-bigvgan-v2"):
        from functools import partial
        from bigvgan.meldataset import mel_spectrogram as _bv_mel
        bigvgan_mel_spectrogram = partial(
            _bv_mel, n_fft=2048,
            num_mels=getattr(hps.model, "bigvgan_mel_channels", 128),
            sampling_rate=44100, hop_size=512, win_size=2048, fmin=0, fmax=None,
        )

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    optim_g.zero_grad(set_to_none=True)
    optim_d.zero_grad(set_to_none=True)
    for batch_idx, items in enumerate(train_loader):
        group_start = (batch_idx // accumulation_steps) * accumulation_steps
        group_end = min(group_start + accumulation_steps, len(train_loader))
        current_accumulation_steps = group_end - group_start
        should_step = batch_idx + 1 == group_end
        should_log = rank == 0 and should_step and global_step % hps.train.log_interval == 0
        cur_lr = get_lr(global_step, base_lr, warmup_steps, decay_steps)

        # 更新主干 lr(第一个 param group),保留 vocoder 的固定 lr(如果有第二个 group)
        optim_g.param_groups[0]['lr'] = cur_lr
        if len(optim_g.param_groups) > 1:
            # Phase2 vocoder 保持固定 lr,不跟 cosine 衰减
            pass  # vocoder lr 已在创建时设置,不覆盖

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

        # no-GAN warmup 是显式配置项；默认不开，避免判别器在某个 step 突然全量接入。
        in_warmup = global_step < disc_warmup_steps
        extra_disc_active = global_step >= disc_start_step

        if is_combined_disc:
            active = set(net_d.kinds) if extra_disc_active else {"mpd"}
            disc_kwargs = {"active": active}
        else:
            disc_kwargs = {}

        sync_gen = nullcontext()
        if not should_step and hasattr(net_g, "no_sync"):
            sync_gen = net_g.no_sync()

        with sync_gen:
            with autocast('cuda', enabled=hps.train.fp16_run, dtype=half_type):
                y_hat, ids_slice, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0, speaker_adv_logits, pred_mel = net_g(c, f0, uv, spec, g=g, c_lengths=lengths,
                                                                                                        spec_lengths=lengths,vol = volume)

                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                # 80-band y_hat_mel:非 BigVGAN 时是每步 loss;BigVGAN 时仅 log 步用于分频段监控/画图,
                # 其余步跳过这次 STFT(loss 走 128-band 的 y_hat_bvmel)。
                need_yhat_mel = bigvgan_mel_spectrogram is None or should_log
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                ) if need_yhat_mel else None
                y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            if not in_warmup:
                sync_disc = nullcontext()
                if not should_step and hasattr(net_d, "no_sync"):
                    sync_disc = net_d.no_sync()
                with sync_disc:
                    with autocast('cuda', enabled=hps.train.fp16_run, dtype=half_type):
                        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach(), **disc_kwargs)
                        with autocast('cuda', enabled=False, dtype=half_type):
                            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                            loss_disc_all = loss_disc
                    scaler.scale(loss_disc_all / current_accumulation_steps).backward()
            else:
                loss_disc = torch.tensor(0.0, device=y_mel.device)
                loss_disc_all = loss_disc
                grad_norm_d = None

            if not in_warmup:
                set_requires_grad(net_d, False)
            try:
                with autocast('cuda', enabled=hps.train.fp16_run, dtype=half_type):
                    # Generator
                    if not in_warmup:
                        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat, **disc_kwargs)
                    with autocast('cuda', enabled=False, dtype=half_type):
                        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                        if not in_warmup:
                            loss_fm = feature_loss(fmap_r, fmap_g)
                            loss_gen, losses_gen = generator_loss(y_d_hat_g)
                        else:
                            loss_fm = torch.tensor(0.0, device=y_mel.device)
                            loss_gen = torch.tensor(0.0, device=y_mel.device)
                        loss_lf0 = F.mse_loss(pred_lf0, lf0) if getattr(net_g, "module", net_g).use_automatic_f0_prediction else 0
                        loss_speaker_adv = F.cross_entropy(speaker_adv_logits, g.squeeze(1)) * c_speaker_adv if speaker_adv_logits is not None else 0
                        # Mel 监督:BigVGAN 系统一到官方 128-band mel 流形,只有一个 target,消除 80/128 双基错位。
                        #   loss_mel(wave-domain): BigVGANMel(y_hat) vs target —— 过 vocoder,是唯一能训 NSF source 的 mel 监督
                        #   loss_bigvgan_mel(direct): pred_mel(mel head 输出) vs target —— 直接拉到 vocoder 输入流形
                        # 非 BigVGAN 声码器退回项目 80-band wave-domain loss。
                        loss_bigvgan_mel = torch.tensor(0.0, device=y_mel.device)
                        if bigvgan_mel_spectrogram is not None:
                            target_mel = bigvgan_mel_spectrogram(y.squeeze(1).float())
                            if use_mel_loss:
                                y_hat_bvmel = bigvgan_mel_spectrogram(y_hat.squeeze(1).float())
                                loss_mel = F.l1_loss(y_hat_bvmel, target_mel) * hps.train.c_mel
                            else:
                                loss_mel = torch.tensor(0.0, device=y_mel.device)
                            if pred_mel is not None and use_bigvgan_mel_loss:
                                loss_bigvgan_mel = F.l1_loss(pred_mel.float(), target_mel) * c_bigvgan_mel
                                # 追踪原始 L1 用于 Phase2 切换判断
                                if bv_strategy and not bv_strategy.in_phase2:
                                    recent_mel_losses.append(loss_bigvgan_mel.item() / c_bigvgan_mel)
                        else:
                            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel if use_mel_loss else torch.tensor(0.0, device=y_mel.device)
                        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0 + loss_speaker_adv + loss_bigvgan_mel
            finally:
                if not in_warmup:
                    set_requires_grad(net_d, True)

            scaler.scale(loss_gen_all / current_accumulation_steps).backward()

        if should_step:
            if not in_warmup:
                scaler.unscale_(optim_d)
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), grad_clip_norm)
                scaler.step(optim_d)
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), grad_clip_norm)
            scaler.step(optim_g)
            scaler.update()
            optim_g.zero_grad(set_to_none=True)
            optim_d.zero_grad(set_to_none=True)
        else:
            grad_norm_g = None

        if should_step and rank == 0:
            if should_log:
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

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr}
                if grad_norm_d is not None and grad_norm_g is not None:
                    scalar_dict.update({"grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g})
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/mel_raw": mel_raw,
                                    "loss/g/kl": loss_kl, "loss/g/lf0": loss_lf0, "loss/g/speaker_adv": loss_speaker_adv,
                                    "loss/g/bigvgan_mel": loss_bigvgan_mel, "loss/g/reference": reference_loss})

                # 高频 / 分频段 mel 监控:嘶声多源于高频带重建误差,分段 L1 可定位问题频带。
                with torch.no_grad():
                    n_bins = y_mel.shape[1]
                    lo, hi = n_bins // 3, 2 * n_bins // 3
                    scalar_dict.update({
                        "mel_band/low_l1": F.l1_loss(y_hat_mel[:, :lo], y_mel[:, :lo]),
                        "mel_band/mid_l1": F.l1_loss(y_hat_mel[:, lo:hi], y_mel[:, lo:hi]),
                        "mel_band/high_l1": F.l1_loss(y_hat_mel[:, hi:], y_mel[:, hi:]),
                    })

                # NSF source 监控:harmonic source RMS + 每层注入比例,
                # 用来判断电音感是否来自 source 注入过强。
                vocoder = get_vocoder_for_stats(net_g)
                src_stats = getattr(vocoder, "source_stats", None) if vocoder is not None else None
                if src_stats:
                    scalar_dict.update({f"source/{k}": v for k, v in src_stats.items()})

                scalar_dict["train/disc_warmup"] = 1.0 if in_warmup else 0.0
                scalar_dict["train/extra_disc_active"] = 1.0 if extra_disc_active else 0.0

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

        # 所有 rank 都要参与 Phase2 决策同步,避免多卡时只有 rank0 解冻/重建 optimizer。
        if should_step and bv_strategy and global_step % hps.train.eval_interval == 0:
            should_enter_phase2 = rank == 0 and bv_strategy.should_transition_to_phase2(global_step, recent_mel_losses)
            if bv_strategy.sync_phase2_decision(should_enter_phase2):
                if rank == 0:
                    logger.info("=" * 70)
                    logger.info(f"[BigVGAN Phase2] MelHead 已收敛(mel L1 < {bv_strategy.phase1_mel_target:.3f}),自动切换到 Phase2:")
                    logger.info(f"  - 解冻 BigVGAN vocoder,微调 lr={bv_strategy.phase2_vocoder_lr}")
                    logger.info(f"  - 开启所有已启用的判别器(disc_start_step={global_step})")
                    logger.info("  - 重建 Optimizer")
                    logger.info("=" * 70)

                optim_g = build_bigvgan_phase2_optimizer(net_g, hps, bv_strategy.phase2_vocoder_lr)
                optims_ref["g"] = optim_g
                disc_start_step = global_step

                if rank == 0:
                    bv_strategy.mark_phase2()

                    utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                          os.path.join(hps.model_dir, f"G_{global_step}_phase2.pth"))
                    logger.info(f"[BigVGAN Phase2] 已存 Phase2 起点 checkpoint: G_{global_step}_phase2.pth")
                else:
                    bv_strategy.in_phase2 = True

        if should_step:
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
    scalar_dict = {}
    speaker_similarity_pairs = []
    speaker_similarity_enabled = getattr(hps.train, "eval_speaker_similarity", True)
    speaker_similarity_items = getattr(hps.train, "eval_speaker_similarity_items", 4)
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
            if speaker_similarity_enabled and len(speaker_similarity_pairs) < speaker_similarity_items:
                speaker_similarity_pairs.append((y_hat[0].detach().cpu(), y[0].detach().cpu()))
        image_dict.update({
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })
    if speaker_similarity_pairs:
        metric = get_speaker_similarity_metric(hps)
        scalar_dict.update(collect_speaker_similarity_scalars(
            speaker_similarity_pairs,
            sample_rate=hps.data.sampling_rate,
            metric=metric,
        ))
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        scalars=scalar_dict,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()

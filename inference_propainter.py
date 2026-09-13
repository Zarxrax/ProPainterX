# -*- coding: utf-8 -*-
import os
import sys
import time

# Reduce CUDA memory fragmentation (PyTorch >= 2.1) for this script's many
# variable-sized chunked allocations. Must be set before the CUDA caching
# allocator initializes, so this needs to happen before `torch` is imported.
# Only takes effect if the user hasn't already set this themselves.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import cv2
import argparse
import imageio
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm

import av
import torch
import torchvision

from model.modules.flow_comp_raft import RAFT_bi
from model.modules.flow_comp_memfof import MEMFOF_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from model.misc import get_device

import warnings
warnings.filterwarnings("ignore")


def _install_oom_diagnostic_hook():
    """Install a sys.excepthook that prints whatever VRAM data is available
    at crash time -- including a live read of the stage that was actually
    in progress -- when the crash looks like a CUDA out-of-memory error.

    Without this, a crash mid-pipeline gives zero VRAM diagnostics: the
    --vram_report summary only ever prints on successful completion, so
    the one run you'd most want data from (the one that OOMs) is exactly
    the one that reports nothing. This works regardless of whether
    --vram_report was passed -- with it, you get the full per-stage
    breakdown for whatever completed plus a live reading for the stage
    that crashed; without it, you still get the raw overall peak PyTorch
    was tracking anyway, which costs nothing to report.

    Reads `device`, `args`, `vram_stats`, `true_vram_stats`, `gpu_poller`
    from the module's global namespace lazily (only at crash time, via
    normal Python name lookup), so this can be installed before any of
    those exist yet -- they only need to exist by the time a crash
    actually happens, which is guaranteed since nothing can crash before
    it runs.
    """
    def _hook(exc_type, exc_value, exc_tb):
        msg = str(exc_value).lower()
        is_oom = 'out of memory' in msg or isinstance(exc_value, getattr(torch.cuda, 'OutOfMemoryError', ()))
        g = globals()
        if is_oom and 'device' in g and g['device'].type == 'cuda':
            print('\n' + '=' * 78)
            print('CUDA OUT OF MEMORY -- VRAM diagnostics at crash time')
            print('=' * 78)
            device = g['device']
            vram_stats = g.get('vram_stats', {})
            true_vram_stats = g.get('true_vram_stats', {})
            if vram_stats:
                print('Stages that completed before the crash:')
                for label, peak in vram_stats.items():
                    if label not in true_vram_stats:
                        print(f'  {label:35s} allocator {peak:9.1f} MB')
                    else:
                        print(f'  {label:35s} allocator {peak:9.1f} MB   true {true_vram_stats[label]:9.1f} MB')
            try:
                # Whatever stage was running when the crash happened never
                # got to call _report_stage(), so pull one final reading
                # directly -- this is the number you actually want for a
                # crash, and is otherwise not printed anywhere.
                in_progress_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
                print(f'  {"(in-progress stage, up to crash)":35s} allocator {in_progress_alloc:9.1f} MB', end='')
                gpu_poller = g.get('gpu_poller')
                if gpu_poller is not None and gpu_poller.is_available:
                    print(f'   true {gpu_poller.peak_mb():9.1f} MB')
                else:
                    print('   (true reading unavailable: install nvidia-ml-py for this)')
            except Exception:
                pass
            print('=' * 78 + '\n')
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = _hook


_install_oom_diagnostic_hook()

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


# resize frames
def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames, process_size, out_size


def pil_list_to_uint8_tensor(imgs, binarize=False):
    """Stack a list of PIL Images into a uint8 CPU tensor (1, T, C, H, W).

    Frames/masks are kept as uint8 while CPU-resident to minimize RAM; each
    chunk is converted to float only once it's on the active device (at each
    call site that does `.to(device)`).

    With binarize=True (for masks, which are always exactly 0 or 255), the
    result is collapsed to {0, 1} directly, so callers needing a float mask
    can just call .float() with no division, and callers needing a raw 0/1
    uint8 mask can use it as-is.
    """
    arr = np.stack([np.array(img) for img in imgs], axis=0)
    if arr.ndim == 3:
        arr = arr[..., None]  # add channel dim for single-channel (mask) images
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (T, C, H, W), uint8
    if binarize:
        t = (t > 127).to(torch.uint8)
    return t.unsqueeze(0)  # (1, T, C, H, W)


def _read_video_pyav(filename):
    """Decode a video file to a list of RGB uint8 HWC frames + fps, using PyAV."""
    container = av.open(filename)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate is not None else 24.0
    frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(stream)]
    container.close()
    return frames, fps


#  read frames from video
def read_frame_from_videos(frame_root):
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, fps = _read_video_pyav(frame_root) # RGB
        frames = [Image.fromarray(f) for f in vframes]
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
    size = frames[0].size

    return frames, fps, size, video_name


def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask
  
  
# read frame-wise masks
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    
    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
       masks_img = [Image.open(mpath)]
    elif mpath.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input mask video path
        mask_vframes, _ = _read_video_pyav(mpath)
        masks_img = [Image.fromarray(f) for f in mask_vframes]
        if len(masks_img) != length:
            print(f'[mask] Warning: mask video has {len(masks_img)} frames, input video has '
                  f'{length} -- these should match.')
    else:  
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))
          
    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    
    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting.
    """
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    # Defines new FOV.
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Extrapolates the FOV for video.
    frames = []
    for v in video_ori:
        frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []
    
    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)
    
    mask[H_start+dilate_h: H_start+imgH-dilate_h, 
         W_start+dilate_w: W_start+imgW-dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start: H_start+imgH, W_start: W_start+imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))
  
    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame
    
    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


def to_device_tuple(t, device):
    return (t[0].to(device), t[1].to(device))


def _vram_reset(device):
    """Zero the peak-memory high-water mark so the next _vram_peak_mb() call
    reports only what happens between this reset and that call."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def _vram_peak_mb(device):
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return None  # no CUDA memory to report (e.g. running on CPU)


def _vram_reserved_peak_mb(device):
    """Peak of PyTorch's own caching-allocator pool (tensors *plus* whatever
    cached-but-currently-unused blocks the allocator is holding onto for
    reuse). Always >= _vram_peak_mb(). The gap between the two is
    fragmentation/headroom PyTorch itself knows about; anything still
    unexplained after that (true nvidia-smi peak minus this number) is
    memory PyTorch has no visibility into at all -- typically cuDNN/cuBLAS
    workspace buffers, allocated via direct cudaMalloc calls that bypass
    the caching allocator entirely."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_reserved(device) / 1024**2
    return None


class _GpuTruePeakPoller:
    """Background poller for the device's *true* memory usage (matches
    nvidia-smi exactly), sampled on a thread via pynvml.

    torch.cuda.max_memory_allocated() -- used by _vram_peak_mb() above --
    only tracks PyTorch's own tensor allocator. It misses the CUDA context
    itself (driver structures, cuBLAS/cuDNN handles: typically 300MB-1GB,
    paid once on the first CUDA call), cuDNN's lazily-allocated workspace
    buffers, and anything else on the device outside PyTorch's bookkeeping.
    A live poll of the whole device is the only way to catch these as a
    real measured peak rather than a one-off snapshot.

    Gracefully disables itself (is_available=False) if pynvml isn't
    installed, there's no NVIDIA driver, or anything else goes wrong --
    every other code path in this script works identically either way.
    """

    def __init__(self, device, interval=0.02, enabled=True):
        self.is_available = False
        self.device = device
        self.interval = interval
        self._peak = 0
        self._stop_event = None
        self._thread = None
        self._pynvml = None
        self._handle = None
        if device.type != 'cuda' or not enabled:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            idx = device.index if device.index is not None else 0
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self._pynvml = pynvml
            self.is_available = True
        except Exception:
            self.is_available = False

    def _sample(self):
        try:
            used = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
            if used > self._peak:
                self._peak = used
        except Exception:
            pass

    def _poll_loop(self):
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self.interval)

    def reset(self):
        """Stop any previous polling, zero the peak, and start polling
        fresh -- call this at the start of each stage you want measured."""
        if not self.is_available:
            return
        import threading
        self.stop()
        self._peak = 0
        self._sample()  # baseline reading in case the stage is very short
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=1.0)
            self._thread = None

    def peak_mb(self):
        if not self.is_available:
            return None
        self._sample()  # catch anything since the last periodic sample
        return self._peak / 1024**2


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--video', type=str, default='inputs/object_removal/bmx-trees', help='Path of the input video or image folder.')
    parser.add_argument(
        '-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask',
        help='Path of the mask(s): a single image, a folder of per-frame mask images, or a video '
             'file of binary mask frames (decoded the same way as --video).')
    parser.add_argument(
        '-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument(
        "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument(
        '--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument(
        '--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument(
        '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument(
        "--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument(
        "--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument(
        "--subvideo_length", type=int, default=80,
        help="Threshold (in frames) for when a video is considered 'long' enough to cap the "
             "global reference-frame budget (see --max_ref_frames) instead of letting it grow "
             "unbounded with video length. Also the default basis for --max_ref_frames and "
             "--img_propagation_chunk_size when those are left unset.")
    parser.add_argument(
        '--max_ref_frames', type=int, default=None,
        help="Maximum number of global reference frames sampled across the whole video, once "
             "video_length exceeds --subvideo_length. A quality/algorithm parameter -- more "
             "reference frames means more long-range temporal consistency, at the cost of more "
             "compute. Left unset, defaults to --subvideo_length // --ref_stride.")
    parser.add_argument(
        '--img_propagation_chunk_size', type=int, default=80,
        help="Number of frames Stage 3a (image propagation) processes per call. Raise for more "
             "temporal context, lower for less VRAM. Not bit-exact regardless of size: image "
             "propagation is a recurrent bidirectional pass (each frame's warped features depend "
             "on the previous frame's), so a chunk boundary resets that context, softened only by "
             "a padding overlap.")
    parser.add_argument(
        '--flow_completion_chunk_size', type=int, default=40,
        help="Number of frames the flow-completion network processes per call. Not guaranteed "
             "bit-exact regardless of size, unlike RAFT/MemFOF's pairwise chunking or the encoder "
             "chunking below: the recurrent hidden state resets at each chunk boundary (softened "
             "by a few frames of padding overlap), so a smaller value trades some temporal context "
             "for memory. Raise for more context, lower for less VRAM.")
    parser.add_argument(
        "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument(
        '--flow_model', default='memfof', choices=['raft', 'memfof'],
        help="Optical flow model to use. 'raft' is the original ProPainter model. "
             "'memfof' (https://github.com/msu-video-group/memfof) is a more memory-efficient, "
             "multi-frame flow estimator built for native 1080p processing -- its all-pairs-free "
             "correlation avoids RAFT's O((H*W)^2) correlation-volume memory blowup at high "
             "resolution. Requires the 'memfof' pip package and downloads its own pretrained "
             "checkpoint from HuggingFace Hub on first use.")
    parser.add_argument(
        '--memfof_model', type=str, default='egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH',
        help="HuggingFace Hub checkpoint to use when --flow_model=memfof. Default is the "
             "checkpoint MemFOF's authors recommend for real-world video.")
    parser.add_argument(
        '--memfof_chunk_len', type=int, default=8,
        help="Number of frames per chunk when --flow_model=memfof. Larger values batch more "
             "windows per call (better GPU throughput, more VRAM) -- the main memory/speed lever "
             "for Stage 1. Raise it if Stage 1's VRAM usage in --vram_report is well below your "
             "other stages; lower it if it climbs sharply or OOMs.")
    parser.add_argument(
        '--memfof_no_shared_fnet', action='store_true',
        help="Disable computing each unique frame's feature map once and reusing it across every "
             "window that needs it (on by default; bit-exact, ~19%% faster, no VRAM downside). "
             "This reaches into MemFOF's internal submodules rather than its public API, so a "
             "future MemFOF version could break it -- use this flag to fall back to the safe, "
             "officially-supported path if that happens.")
    parser.add_argument(
        '--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'], help="Modes: video_inpainting / video_outpainting")
    parser.add_argument(
        '--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
    parser.add_argument(
        '--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
    parser.add_argument(
        '--save_fps', type=int, default=24, help='Frame per second. Default: 24')
    parser.add_argument(
        '--save_frames', action='store_true', help='Save output frames. Default: False')
    parser.add_argument(
        '--no-fp16', dest='fp16', action='store_false', default=True,
        help='Disable fp16 (half precision) and run in fp32 (single precision) instead. '
             'fp16 is on by default.')
    parser.add_argument(
        '--encoder_chunk_size', type=int, default=10,
        help="Number of frames to run through ProPainter's per-frame-independent components "
             "(encoder, SoftSplit/SoftComp -- the single largest transient tensors in the model, "
             "since each token's raw unfolded form is ~12x larger than its embedded form -- and "
             "the decoder) at a time, within a single transformer-stage window. Also reused for "
             "Stage 2's flow-completion decoder (decoder2/decoder1/upsample), which has the exact "
             "same zero-cross-frame-mixing property once its features are reshaped to a per-frame "
             "batch. None of these mix information across frames, so this is a pure memory/"
             "scheduling optimization with bit-exact output -- unlike --neighbor_length below, it "
             "does not trade any quality for memory. Lower this first if you're VRAM-constrained. "
             "Set to a large number (e.g. 9999) to disable and process the whole window in one "
             "shot (the original behavior).")
    parser.add_argument(
        '--auto_high_res_settings', action='store_true',
        help="If set, automatically caps --neighbor_length downward at high resolutions to control "
             "the transformer stage's VRAM use (mirrors the existing resolution-based auto-tuning "
             "already used for RAFT's short_clip_len). Unlike --encoder_chunk_size, this DOES trade "
             "some temporal context/quality for memory, so it is off by default; a message is "
             "printed whenever it actually changes a setting.")
    parser.add_argument(
        '--vram_report', action='store_true',
        help='Print peak GPU memory used by each pipeline stage (optical flow, flow completion, '
             'ProPainter) and a final summary. No effect (reports "N/A") when running on CPU.')
    parser.add_argument(
        '--native_resolution_composite', action='store_true',
        help="If set (and --mode is video_inpainting), the final output is composited at the "
             "video's true native resolution instead of whatever (possibly downscaled) resolution "
             "was used for processing. Only the masked region comes from the network's prediction "
             "(upscaled); everything outside a feathered mask boundary is the untouched original "
             "frame at full quality. This lets you downscale heavily for VRAM/speed (--resize_ratio, "
             "--width/--height) without visibly softening the majority of the frame that was never "
             "part of the removed object -- only the repainted region itself will show some "
             "softness from being upscaled. No effect if you're already processing at native "
             "resolution. Not supported in video_outpainting mode.")
    parser.add_argument(
        '--composite_feather_px', type=int, default=15,
        help='Gaussian blur radius (in native-resolution pixels) for the mask boundary when '
             '--native_resolution_composite is used, to avoid a visible sharp/blurry seam where '
             'the upscaled prediction meets the untouched original. 0 = hard edge.')

    args = parser.parse_args()

    vram_stats = {}          # PyTorch allocator peak (tensors only) per stage
    reserved_vram_stats = {} # PyTorch caching-allocator pool peak (tensors + its own idle/cached blocks) per stage
    true_vram_stats = {}     # true device-memory peak per stage, via pynvml (optional)
    time_stats = {}          # wall-clock seconds per stage
    gpu_poller = _GpuTruePeakPoller(device, enabled=args.vram_report)
    if args.vram_report and device.type == 'cuda' and not gpu_poller.is_available:
        print('[VRAM] Note: install `nvidia-ml-py` (pip) for a second, more complete reading '
              'that also captures CUDA context/cuDNN overhead outside PyTorch\'s own allocator. '
              'Continuing with PyTorch allocator stats only.')

    _stage_start_time = [None]  # boxed so the nested functions below can rebind it

    def _stage_reset():
        """Call at the start of a stage you want to measure."""
        if not args.vram_report:
            return
        # GPU work is asynchronous -- without this, a slow stage's tail end
        # could still be running when we start timing the *next* stage,
        # silently shifting time from one stage's number into another's.
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        _vram_reset(device)
        gpu_poller.reset()
        _stage_start_time[0] = time.perf_counter()

    def _report_stage(label):
        """Record + print the peak GPU memory and elapsed wall-clock time
        since the last _stage_reset() call, under the given stage label.
        Reports two memory numbers when pynvml is available: PyTorch's own
        allocator peak (tensors only), and the true device-memory peak
        (matches nvidia-smi, includes CUDA context/cuDNN overhead)."""
        if not args.vram_report:
            return
        if device.type == 'cuda':
            torch.cuda.synchronize(device)  # same reason as in _stage_reset()
        elapsed = time.perf_counter() - _stage_start_time[0]
        time_stats[label] = elapsed
        peak = _vram_peak_mb(device)
        vram_stats[label] = peak
        reserved_peak = _vram_reserved_peak_mb(device)
        if reserved_peak is not None:
            reserved_vram_stats[label] = reserved_peak
        true_peak = gpu_poller.peak_mb()
        if true_peak is not None:
            true_vram_stats[label] = true_peak
        if peak is not None:
            if true_peak is not None:
                print(f'[VRAM] {label:35s} allocator {peak:9.1f} MB   reserved {reserved_peak:9.1f} MB   true (nvidia-smi) {true_peak:9.1f} MB   time {elapsed:7.1f}s')
            else:
                print(f'[VRAM] {label:35s} peak {peak:9.1f} MB   reserved {reserved_peak:9.1f} MB   time {elapsed:7.1f}s')
        else:
            print(f'[VRAM] {label:35s} N/A (running on CPU)   time {elapsed:7.1f}s')

    if args.vram_report:
        # One-time "Stage 0": force CUDA context creation now (rather than
        # letting it happen invisibly inside Stage 1) and measure it, since
        # this fixed tax is otherwise never attributed to anything -- it's
        # not part of PyTorch's own allocator stats at all.
        _stage0_start = time.perf_counter()
        gpu_poller.reset()
        if device.type == 'cuda':
            _ = torch.zeros(1, device=device)
            torch.cuda.synchronize(device)
            del _
        ctx_peak = gpu_poller.peak_mb()
        ctx_reserved = torch.cuda.memory_reserved(device) / 1024**2 if device.type == 'cuda' else None
        ctx_elapsed = time.perf_counter() - _stage0_start
        time_stats['Stage 0: CUDA context init'] = ctx_elapsed
        if ctx_reserved is not None:
            vram_stats['Stage 0: CUDA context init'] = ctx_reserved
            reserved_vram_stats['Stage 0: CUDA context init'] = ctx_reserved
            if ctx_peak is not None:
                true_vram_stats['Stage 0: CUDA context init'] = ctx_peak
                print(f'[VRAM] {"Stage 0: CUDA context init":35s} allocator {ctx_reserved:9.1f} MB   reserved {ctx_reserved:9.1f} MB   true (nvidia-smi) {ctx_peak:9.1f} MB   time {ctx_elapsed:7.1f}s')
            else:
                print(f'[VRAM] {"Stage 0: CUDA context init":35s} reserved {ctx_reserved:9.1f} MB   time {ctx_elapsed:7.1f}s')
        else:
            print(f'[VRAM] {"Stage 0: CUDA context init":35s} N/A (running on CPU)   time {ctx_elapsed:7.1f}s')

    _stage_reset()

    # Use fp16 precision during inference to reduce running memory cost
    use_half = True if args.fp16 else False 
    if device == torch.device('cpu'):
        use_half = False

    frames, fps, size, video_name = read_frame_from_videos(args.video)
    # Keep a reference to the frames at their true native resolution, but
    # only when --native_resolution_composite might use them (the only place
    # frames_native_pil is read) -- otherwise this would hold a second
    # full-resolution copy of the whole video for no purpose. `frames` gets
    # reassigned to a (possibly downscaled) copy below by resize_frames(),
    # but PIL's .resize() returns a new image rather than mutating in place,
    # so this list of the original objects (when kept) stays untouched by
    # any processing-resolution downscaling.
    frames_native_pil = frames if args.native_resolution_composite else None
    native_size = frames[0].size

    if not args.width == -1 and not args.height == -1:
        size = (args.width, args.height)
    if not args.resize_ratio == 1.0:
        size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))

    frames, size, out_size = resize_frames(frames, size)
    
    fps = args.save_fps if fps is None else fps
    save_root = os.path.join(args.output, video_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    if args.mode == 'video_inpainting':
        frames_len = len(frames)
        flow_masks, masks_dilated = read_mask(args.mask, frames_len, size, 
                                              flow_mask_dilates=args.mask_dilation,
                                              mask_dilates=args.mask_dilation)
        w, h = size
    elif args.mode == 'video_outpainting':
        assert args.scale_h is not None and args.scale_w is not None, 'Please provide a outpainting scale (s_h, s_w).'
        frames, flow_masks, masks_dilated, size = extrapolation(frames, (args.scale_h, args.scale_w))
        w, h = size
    else:
        raise NotImplementedError

    if args.auto_high_res_settings:
        # Mirrors the resolution-based auto-tuning already used for RAFT's
        # short_clip_len, but applied to the transformer stage's frame
        # window. Unlike short_clip_len (a pure scheduling parameter with no
        # effect on output), --neighbor_length controls how much temporal
        # context the transformer sees, so this is a genuine quality/memory
        # trade-off -- only ever caps downward, never raises a value the
        # user explicitly set lower, and always prints what it changed.
        if max(w, h) > 1280:
            neighbor_cap = 6
        elif max(w, h) > 720:
            neighbor_cap = 8
        else:
            neighbor_cap = None
        if neighbor_cap is not None and args.neighbor_length > neighbor_cap:
            print(f'[auto_high_res_settings] resolution {w}x{h}: capping --neighbor_length '
                  f'{args.neighbor_length} -> {neighbor_cap} to control transformer VRAM use. '
                  f'This reduces local temporal context; pass --neighbor_length explicitly '
                  f'(without --auto_high_res_settings) to override.')
            args.neighbor_length = neighbor_cap
    
    # For saving the masked-input preview video. Streamed straight to disk as
    # each frame is computed rather than accumulated in RAM, since all its
    # inputs (frames, masks_dilated) are already available here, well before
    # any GPU stage starts.
    masked_in_writer = imageio.get_writer(os.path.join(save_root, 'masked_in.mp4'), fps=fps, quality=7)
    for i in range(len(frames)):
        mask_ = np.expand_dims(np.array(masks_dilated[i]),2).repeat(3, axis=2)/255.
        img = np.array(frames[i])
        green = np.zeros([h, w, 3]) 
        green[:,:,1] = 255
        alpha = 0.6
        # alpha = 1.0
        fuse_img = (1-alpha)*img + alpha*green
        fuse_img = mask_ * fuse_img + (1-mask_)*img
        fuse_img = cv2.resize(fuse_img.astype(np.uint8), out_size)
        masked_in_writer.append_data(fuse_img)
    masked_in_writer.close()

    # Stored as uint8 (frames: raw [0,255]; masks: collapsed to {0,1}) rather
    # than eagerly converting the whole video to float32 here -- see
    # pil_list_to_uint8_tensor's docstring. Each chunk gets converted to
    # float only once it's already on the active device, at each call site
    # below.
    frames = pil_list_to_uint8_tensor(frames)                          # (1, T, 3, H, W) uint8, [0,255]
    frames_inp = frames[0].permute(0, 2, 3, 1).numpy()                 # (T, H, W, 3) uint8 -- same data, no separate PIL pass
    flow_masks = pil_list_to_uint8_tensor(flow_masks, binarize=True)   # (1, T, 1, H, W) uint8, {0,1}
    masks_dilated = pil_list_to_uint8_tensor(masks_dilated, binarize=True)  # (1, T, 1, H, W) uint8, {0,1}

    # ---- VRAM strategy ----
    # `frames`, `flow_masks` and `masks_dilated` hold the entire video and
    # stay resident on CPU/host RAM for the whole run. Only the sub-chunk
    # currently being processed is moved to `device`, with results copied
    # back to the CPU immediately afterwards (with explicit `del` +
    # `torch.cuda.empty_cache()`). This bounds peak VRAM to roughly one
    # model's weights plus one chunk's activations. The three models
    # (RAFT/MemFOF, flow-completion, ProPainter) are constructed on-demand
    # and freed as soon as their stage finishes, rather than all being
    # resident on the GPU simultaneously.
    video_length = frames.size(1)
    print(f'\nProcessing: {video_name} [{video_length} frames]...')

    ##############################################
    # ---- stage 1: optical flow ----
    ##############################################
    if args.flow_model == 'memfof':
        # MemFOF downloads its own pretrained checkpoint from HuggingFace Hub
        # on first use (via MEMFOF.from_pretrained inside MEMFOF_bi).
        use_shared_fnet = not args.memfof_no_shared_fnet
        fix_raft = MEMFOF_bi(args.memfof_model, device, use_shared_fnet=use_shared_fnet)
        # 6 refinement iterations is well-supported for MemFOF (4->6 gives a
        # real improvement, 6->8 is negligible); used unless --raft_iter is
        # set explicitly (i.e. left at RAFT's default of 20).
        flow_iters = 6 if args.raft_iter == 20 else args.raft_iter
    else:
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                        model_dir='weights', progress=True, file_name=None)
        fix_raft = RAFT_bi(ckpt_path, device)
        flow_iters = args.raft_iter

    # RAFT's correlation-volume computation is numerically unstable in fp16,
    # so it always runs in fp32 regardless of fp16 being on. MemFOF doesn't
    # share that instability and benefits from autocast like every other
    # stage, so it's the only one of the two gated by use_half.
    stage1_autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16,
                                          enabled=(use_half and args.flow_model == 'memfof'))

    with torch.inference_mode(), stage1_autocast_ctx:
        if args.flow_model == 'memfof':
            # MemFOF doesn't have RAFT's all-pairs correlation volume (the
            # O((H*W)^2) blowup that forces RAFT's short_clip_len this low),
            # so it uses its own, less conservative chunk size instead.
            short_clip_len = args.memfof_chunk_len
        elif frames.size(-1) <= 640: 
            short_clip_len = 12
        elif frames.size(-1) <= 720: 
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2
        
        if video_length > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in tqdm(range(0, video_length, short_clip_len), desc=f'Stage 1: optical flow ({args.flow_model})'):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    chunk = frames[:, f:end_f].to(device)
                else:
                    chunk = frames[:, f-1:end_f].to(device)
                # frames is stored as uint8 [0,255] CPU-side (see
                # pil_list_to_uint8_tensor); both RAFT_bi and MEMFOF_bi expect
                # [-1,1] float input, so convert only this chunk, only once
                # it's already on the active device.
                chunk = chunk.float() / 255 * 2 - 1
                flows_f, flows_b = fix_raft(chunk, iters=flow_iters)

                gt_flows_f_list.append(flows_f.cpu())
                gt_flows_b_list.append(flows_b.cpu())
                del chunk, flows_f, flows_b
                torch.cuda.empty_cache()
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
            del gt_flows_f_list, gt_flows_b_list
        else:
            chunk = frames.to(device).float() / 255 * 2 - 1
            flows_f, flows_b = fix_raft(chunk, iters=flow_iters)
            gt_flows_bi = (flows_f.cpu(), flows_b.cpu())
            del chunk, flows_f, flows_b
            torch.cuda.empty_cache()

    del fix_raft
    torch.cuda.empty_cache()
    _report_stage(f'Stage 1: optical flow ({args.flow_model})')
    _stage_reset()

    ##############################################
    # ---- stage 2: flow completion ----
    ##############################################
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path, encoder_chunk_size=args.encoder_chunk_size)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()
    # Weights stay fp32; autocast below handles the fp16 compute, so
    # precision-sensitive ops inside this net aren't forced into fp16 along
    # with everything else.
    stage2_autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_half)

    with torch.inference_mode(), stage2_autocast_ctx:
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > args.flow_completion_chunk_size:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, args.flow_completion_chunk_size):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + args.flow_completion_chunk_size + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + args.flow_completion_chunk_size)

                flows_chunk = (gt_flows_bi[0][:, s_f:e_f].to(device), gt_flows_bi[1][:, s_f:e_f].to(device))
                # masks_dilated/flow_masks are stored as uint8 {0,1} (see
                # pil_list_to_uint8_tensor), so this is just a plain cast, no
                # division needed.
                masks_chunk = flow_masks[:, s_f:e_f+1].to(device).float()

                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(flows_chunk, masks_chunk)
                pred_flows_bi_sub = fix_flow_complete.combine_flow(flows_chunk, pred_flows_bi_sub, masks_chunk)

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e].cpu())
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e].cpu())
                del flows_chunk, masks_chunk, pred_flows_bi_sub
                torch.cuda.empty_cache()
                
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
            del pred_flows_f, pred_flows_b
        else:
            flows_all = to_device_tuple(gt_flows_bi, device)
            masks_all = flow_masks.to(device).float()
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(flows_all, masks_all)
            pred_flows_bi = fix_flow_complete.combine_flow(flows_all, pred_flows_bi, masks_all)
            pred_flows_bi = (pred_flows_bi[0].cpu(), pred_flows_bi[1].cpu())
            del flows_all, masks_all
            torch.cuda.empty_cache()

    del gt_flows_bi, flow_masks, fix_flow_complete
    torch.cuda.empty_cache()
    _report_stage('Stage 2: flow completion')
    _stage_reset()

    ##############################################
    # ---- stage 3: ProPainter (image propagation + transformer) ----
    ##############################################
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path, encoder_chunk_size=args.encoder_chunk_size).to(device)
    model.eval()
    # Model weights stay fp32; autocast below runs the conv/linear/matmul-heavy
    # work (encoder, decoder, SoftSplit/SoftComp, deformable alignment,
    # attention matmuls) in fp16 for speed/VRAM, but keeps precision-sensitive
    # reductions -- LayerNorm and softmax inside SparseWindowAttention -- in
    # fp32, where they're numerically safer. With --no-fp16, autocast is
    # disabled and everything runs in plain fp32.
    autocast_ctx = torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu',
                                   dtype=torch.float16, enabled=use_half)

    with torch.inference_mode(), autocast_ctx:
        # ---- image propagation ----
        # frames/masks_dilated are stored as uint8 CPU-side (see
        # pil_list_to_uint8_tensor); each branch below converts and combines
        # only the chunk it actually needs, right before use.
        # NOTE (see --img_propagation_chunk_size help): image propagation is a
        # genuinely recurrent bidirectional pass, so -- like flow completion above --
        # this chunk size is a real quality/memory tradeoff, not a free bit-exact
        # scheduling choice. The pad_len=10 overlap below softens but doesn't
        # eliminate the context loss at each chunk boundary.
        subvideo_length_img_prop = args.img_propagation_chunk_size
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                flows_chunk = (pred_flows_bi[0][:, s_f:e_f-1].to(device), pred_flows_bi[1][:, s_f:e_f-1].to(device))
                # Converted once here (small: just this chunk) and reused for
                # both the GPU propagation call below and the CPU blend after,
                # instead of converting twice.
                frames_chunk = frames[:, s_f:e_f].float() / 255 * 2 - 1
                masks_chunk_cpu = masks_dilated[:, s_f:e_f].float()
                masked_chunk = (frames_chunk * (1 - masks_chunk_cpu)).to(device)
                masks_chunk = masks_chunk_cpu.to(device)

                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_chunk, flows_chunk, masks_chunk, 'nearest')
                prop_imgs_sub = prop_imgs_sub.cpu()
                updated_local_masks_sub = updated_local_masks_sub.cpu()

                updated_frames_sub = frames_chunk * (1 - masks_chunk_cpu) + \
                                    prop_imgs_sub.view(b, t, 3, h, w) * masks_chunk_cpu
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])

                del flows_chunk, frames_chunk, masks_chunk_cpu, masked_chunk, masks_chunk, prop_imgs_sub, updated_local_masks_sub
                torch.cuda.empty_cache()
                
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            # This branch only ever runs when video_length <= img_propagation_chunk_size
            # (<=80 frames by default), so converting the whole thing to float here
            # is already bounded to a small clip, not the full multi-minute video.
            b, t, _, _, _ = masks_dilated.size()
            flows_all = to_device_tuple(pred_flows_bi, device)
            frames_f = frames.float() / 255 * 2 - 1
            masks_f = masks_dilated.float()
            masked_all = (frames_f * (1 - masks_f)).to(device)
            masks_all = masks_f.to(device)
            prop_imgs, updated_local_masks = model.img_propagation(masked_all, flows_all, masks_all, 'nearest')
            updated_frames = frames_f * (1 - masks_f) + prop_imgs.cpu().view(b, t, 3, h, w) * masks_f
            updated_masks = updated_local_masks.cpu().view(b, t, 1, h, w)
            del flows_all, frames_f, masks_f, masked_all, masks_all, prop_imgs, updated_local_masks
            torch.cuda.empty_cache()

        del frames
        torch.cuda.empty_cache()
        _report_stage('Stage 3a: ProPainter image propagation')
        _stage_reset()

        ori_frames = frames_inp
        comp_frames = [None] * video_length

        neighbor_stride = args.neighbor_length // 2
        # --max_ref_frames controls how many global reference frames are
        # sampled on long videos; when left unset, falls back to
        # --subvideo_length // --ref_stride.
        max_ref_frames = (args.max_ref_frames if args.max_ref_frames is not None
                          else args.subvideo_length // args.ref_stride)
        ref_num = max_ref_frames if video_length > args.subvideo_length else -1
        
        # ---- feature propagation + transformer ----
        for f in tqdm(range(0, video_length, neighbor_stride), desc='Stage 3b: transformer'):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                    min(video_length, f + neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :].to(device)
            # masks_dilated is stored as uint8 {0,1}; plain cast, no division.
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :].to(device).float()
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :].to(device)
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :].to(device),
                                       pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :].to(device))

            # 1.0 indicates mask
            l_t = len(neighbor_ids)

            # pred_img = selected_imgs # results of image propagation
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)

            pred_img = pred_img.cpu().view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].permute(
                0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                    + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else: 
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)

            del selected_imgs, selected_masks, selected_update_masks, selected_pred_flows_bi, pred_img
            torch.cuda.empty_cache()

    _report_stage('Stage 3b: ProPainter transformer (feature prop)')

    del model, pred_flows_bi, updated_frames, updated_masks, masks_dilated
    torch.cuda.empty_cache()

    if args.native_resolution_composite:
        if args.mode != 'video_inpainting':
            print('[native_resolution_composite] Not supported in video_outpainting mode; ignoring.')
        elif native_size == out_size:
            print('[native_resolution_composite] Already processing at native resolution; nothing to do.')
        else:
            print(f'[native_resolution_composite] Compositing predictions into native '
                  f'{native_size[0]}x{native_size[1]} frames (processed at {out_size[0]}x{out_size[1]}).')
            # Re-derive a dilated mask at native resolution. Independent of
            # the processing-resolution mask (already freed above): reads
            # straight from the original mask file(s) again, which is cheap
            # regardless of resolution.
            _, native_masks_dilated = read_mask(args.mask, frames_len, native_size,
                                                 flow_mask_dilates=args.mask_dilation,
                                                 mask_dilates=args.mask_dilation)
            feather_px = args.composite_feather_px
            native_comp_frames = []
            for idx in range(video_length):
                # comp_frames[idx] already has the (processing-resolution)
                # prediction composited into the masked region, with the
                # (processing-resolution, lower quality) original elsewhere.
                # Upscaling the whole thing and then re-compositing against
                # the TRUE native original using a native-resolution mask
                # means only the upscaled *prediction* survives outside of
                # the untouched original -- the upscaled-original portion
                # computed here is entirely discarded below.
                upscaled = cv2.resize(comp_frames[idx], native_size, interpolation=cv2.INTER_CUBIC)

                mask_native = np.array(native_masks_dilated[idx].convert('L'), dtype=np.float32) / 255.0
                if feather_px > 0:
                    k = feather_px * 2 + 1  # odd kernel size required by GaussianBlur
                    mask_native = cv2.GaussianBlur(mask_native, (k, k), 0)
                mask_native = mask_native[:, :, None]  # broadcast over channels

                native_orig = np.array(frames_native_pil[idx]).astype(np.float32)
                blended = upscaled.astype(np.float32) * mask_native + native_orig * (1 - mask_native)
                native_comp_frames.append(blended.astype(np.uint8))

            comp_frames = native_comp_frames
            out_size = native_size
            del native_comp_frames, native_masks_dilated

    # save each frame
    if args.save_frames:
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation = cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img_save_root = os.path.join(save_root, 'frames', str(idx).zfill(4)+'.png')
            imwrite(f, img_save_root)
                    

    # if args.mode == 'video_outpainting':
    #     comp_frames = [i[10:-10,10:-10] for i in comp_frames]

    # save video (masked_in.mp4 is streamed out separately, above)
    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
    imageio.mimwrite(os.path.join(save_root, 'inpaint_out.mp4'), comp_frames, fps=fps, quality=7)
    
    print(f'\nAll results are saved in {save_root}')

    if args.vram_report:
        print('\n' + '=' * 110)
        print('VRAM & timing summary')
        print('=' * 110)
        have_true = len(true_vram_stats) > 0
        print(f'  {"stage":35s} {"allocator (MB)":>16s} {"reserved (MB)":>16s} {"true / nvidia-smi (MB)":>24s} {"time (s)":>10s}')
        for label, elapsed in time_stats.items():
            peak = vram_stats.get(label)
            peak_str = f'{peak:16.1f}' if peak is not None else f'{"n/a":>16s}'
            reserved_str = f'{reserved_vram_stats[label]:16.1f}' if label in reserved_vram_stats else f'{"n/a":>16s}'
            true_str = f'{true_vram_stats[label]:24.1f}' if label in true_vram_stats else f'{"n/a":>24s}'
            print(f'  {label:35s} {peak_str} {reserved_str} {true_str} {elapsed:10.1f}')
        print('-' * 110)
        total_time = sum(time_stats.values())
        print(f'  {"Total time":35s} {"":16s} {"":16s} {"":24s} {total_time:10.1f}')
        if vram_stats and any(v is not None for v in vram_stats.values()):
            overall_peak = max(v for v in vram_stats.values() if v is not None)
            print(f'  {"Overall peak, allocator (largest single stage)":35s} {overall_peak:16.1f}')
            if reserved_vram_stats:
                overall_reserved_peak = max(reserved_vram_stats.values())
                print(f'  {"Overall peak, reserved (PyTorch pool)":35s} {overall_reserved_peak:16.1f}')
            if have_true:
                overall_true_peak = max(true_vram_stats.values())
                print(f'  {"Overall peak, true GPU memory":35s} {overall_true_peak:16.1f} MB  <- use this one for '
                      f'"will it fit on my card"')
        else:
            print('  (No CUDA device in this run -- memory columns are n/a, timing above is still real.)')
        print('=' * 110)

    gpu_poller.stop()
    torch.cuda.empty_cache()

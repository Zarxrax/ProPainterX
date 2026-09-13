## ⚡ About This Fork (ProPainterX)

This is a fork of the original [ProPainter](https://github.com/sczhou/ProPainter) (all credit to the original authors), focused on running faster and fitting in less GPU/CPU memory. Main changes from upstream:

- **Built for consumer hardware**: Uses significantly less VRAM than the original ProPainter, while also being faster, and achieving similar quality.
- **Modernized dependencies**: Runs on current versions of pytorch and other dependencies (as of 2026).
- **MemFOF for optical flow** (`--flow_model memfof`, on by default): an alternative to RAFT built for higher resolutions. RAFT's correlation volume grows quadratically with resolution; MemFOF uses a much smaller local correlation instead, avoiding that blowup at high resolution.
- **Real mixed precision throughout**: every stage runs under `torch.autocast` rather than a blanket `.half()` cast, so precision-sensitive operations (LayerNorm, softmax) stay in fp32 for stability while the rest runs in fp16 for speed and memory.
- **Independent, per-stage chunk sizes**: optical flow, flow completion, image propagation, and the transformer's encoder/decoder each have their own memory/speed dial, instead of one `--subvideo_length` value controlling everything. Several of these are provably lossless (pure scheduling, no effect on output); others trade a small amount of temporal context for memory, and are documented as such in `--help`.
- **CPU-resident pipeline**: frames, masks, and flow are kept on the CPU for the whole run, stored compactly as `uint8` rather than `float32`; only the chunk currently being processed is moved to the GPU, and each of the three models is loaded and freed as its stage finishes.
- **Optimized transformer internals**: fused attention (`scaled_dot_product_attention`) instead of hand-rolled softmax, fused QKV projections, and fewer redundant tensor copies -- all at no cost to output quality.
- **Optional native-resolution output**: process at a lower resolution for speed, then composite the prediction back into the untouched original at full resolution, so only the actually-repainted region shows any softness (`--native_resolution_composite`).
- **`--vram_report`**: prints per-stage peak GPU memory and timing, for diagnosing where memory actually goes on your own clips.

Run `python inference_propainter.py --help` for the full, current list of flags -- several defaults and options differ from upstream. Defaults have been tweaked to be sane for consumer GPUs.

## Updates
- **2026.09.13**: This fork is created.
  
## Results (from original ProPainter)

#### 👨🏻‍🎨 Object Removal
<table>
<tr>
   <td> 
      <img src="assets/object_removal1.gif">
   </td>
   <td> 
      <img src="assets/object_removal2.gif">
   </td>
</tr>
</table>

#### 🎨 Video Completion
<table>
<tr>
   <td> 
      <img src="assets/video_completion1.gif">
   </td>
   <td> 
      <img src="assets/video_completion2.gif">
   </td>
</tr>
<tr>
   <td> 
      <img src="assets/video_completion3.gif">
   </td>
   <td> 
      <img src="assets/video_completion4.gif">
   </td>
</tr>
</table>


## Dependencies and Installation

1. Clone Repo

   ```bash
   git clone https://github.com/Zarxrax/ProPainterX.git
   ```

2. Install Pytorch and other Dependencies

   Install pytorch using the command from https://pytorch.org/

   ```bash
   # install python dependencies
   pip3 install -r requirements.txt
   ```

## Get Started
### Prepare pretrained models
All pretrained models will be automatically downloaded during the first inference.

### 🏂 Quick test
We provide some examples in the [`inputs`](./inputs) folder. 
Run the following commands to try it out:
```shell
# The first example (object removal)
python inference_propainter.py --video inputs/object_removal/bmx-trees --mask inputs/object_removal/bmx-trees_mask 
# The second example (video completion)
python inference_propainter.py --video inputs/video_completion/running_car.mp4 --mask inputs/video_completion/mask_square.png
```

The results will be saved in the `results` folder.
To test your own videos, please prepare the input `mp4 video` (or `split frames`) and `frame-wise mask(s)`.

If you want to specify the video resolution for processing or avoid running out of memory, you can set the video size of `--width` and `--height`. Add `--native_resolution_composite` to composite the inpainted region back onto the full resolution video.

### 🚀 Memory-efficient inference

On the original ProPainter, an 11-second video at 640x480 required 15GB of VRAM and was about 40% slower than ProPainterX, which only needs 5GB for the same clip. 

At default settings, a video at 960x540 resolution (half-HD) can run on an 8GB GPU, and a 1280x720 video can run on a 12GB GPU. Processing full 1080p likely requires a 24GB GPU. 

VRAM is fully decoupled from video length in ProPainterX, though longer videos do require more system RAM, so this project is still not suitable for extremely long clips.

Run `python inference_propainter.py --help` for the full set of tuning flags, and add `--vram_report` to an actual run to see live per-stage memory numbers on your own clips.

## License

#### Non-Commercial Use Only Declaration
The ProPainter is made available for use, reproduction, and distribution strictly for non-commercial purposes. The code and models are licensed under <a rel="license" href="./LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.
# Project Clarifications

---

## Project Philosophy & Goals
What is goal of the project? 
- [Systems] To create an FPV frame interpolation system? --> Why not choose something more SOTA from 2024/25
- [AIGC/ML] To create a new real-time frame interpolation algorithm?
- Answer: The goal was to build on the existing framework of RIFE, which is a widely known framework that works well and is suited to our use case. It was Sanket's decision to use RIFE, so I trust that he picked the best option; in addition, there have been many more recent updates to RIFE (such as practical RIFE, RIFE HDv3, etc.) that may be more SOTA, so it would be good to test that. I don't think the goal is to create an entirely new algorithm, but expand on RIFE for our use case to get rid of distortions and other issues.

Although both are great avenues, it is important to define the type of innovation we are aiming for, since that will allow us to use the relevant tools (Robotics vs AIGC vs Both).
Answer: Of the above, I think our goal matches the creation of an FPV frame interpolation system more.

## What is the typical workflow: 
- frame extraction --> batch_interpolate.py --> rename.py --> video encoding
- Please document this well for general reference (will be needed when we publish the paper)
- Answer: I agree; the current typical workflow is:
  Original 30 FPS frames (original_frames) --> downsample to 10 FPS frames (10FPS_frames) --> batch_interpolate.py with 10FPS_frames to get interpolated 30 FPS frames (interpolated_frames) --> rename.py with interpolated_frames --> video encoding with interpolated_frames, and/or comparison of original_frames with interpolated_frames

## The goal is video_in --> interpolated_video_out. Correct?
- How to achieve that in this repo?
- Answer: Yes, I understand that to be the goal. I'll work on combining the current workflow into one script.

## Please explain the septuplet logic
- Any way to quantify the movement (of camera/objects) in input frames (training vs inference)?
- Answer: There are 7 frames in each septuplet (30 FPS): F1, F2, F3, F4, F5, F6, F7. For our use case, interpolating from 10 to 30 FPS, we want to interpolate 2 additional frames between every frame pair in the 10 FPS frames (one at t=1/3, another at t=2/3). So from each septuplet, we can choose a triplet where the 1st and 3rd frames are separated by 2 frames, and the 2nd frame is one of those 2 intermediate frames (either t=1/3 or t=2/3). For example, F1 F2 F4 (t=1/3), or F4 F6 F7 (t=2/3). This allows us to get many more training samples from each septuplet.

## Please help document the gradiant accumulation?
- Commit 3575169 ["adding gradient accumulation for the 8 triplets per septuplet"]
- Why was this done? Is it a established practice?
- Answer: I wanted to maintain the batch size to keep the learning rate the same; since the model is only stepped once every 4 triplets, the batch size should still be treated as the same. I average the 4 losses before the optimization step. I think it is an established practice, it seems to be used in DAIN (CVPR 2019), and from some reading online, it seems accumulating the gradient across multiple micro-batches and stepping once per large batch has been done. Since I'm not as knowledgeable in the subject, though, it would be very helpful to hear your thoughts on this.

## Record of Hyperparameters & their effects, trained/fine-tuned model checkpoints & metrics?
- List of Hyperparameters and one-line/10 word explanations
- A record of hyperparameters and their effect on Performance metrics 
  - costly experiments, not a good idea to let them get lost
  - Have some way to analyze over/under fitting, training smoothness
- Ability to resume training if required (power/network outage or curriculum training)
- Answer:
- no_DDP: true; only using 1 GPU
- epoch: 200 (was 300, may be limiting model)
- batch size: 16 (see above explanation about batch size)
- learning rate: initial is 3e-4, minimum is 3e-6, 1400 steps before cosine scheduling begins
- number of gradient accumulation steps: 4
- number of workers: 2 for training, 8 for validation
- validation frequency: every 5 epochs
- I will look into allowing the model to resume training.
- Since I've only trained the model once so far, I'm not sure of each's effect on performance metrics yet; it would be helpful to hear some advice on how to do this.

## What are "cycles" and "error" in this context (iterative refinement loops)?
- Commit 786e220 ["increased max inference_img.py cycles to 12, error to 0.0001"]
- Answer: This is outdated now, since we've transitioned from interpolating using a binary recursive method (IFNet) to directly interpolating at t=1/3 or t=2/3. Originally, we had to interpolate at t=0.5 again and again for multiple cycles to approximate t=1/3 or t=2/3 within a certain error.

## Slide 30 in your presentation
`One sample --> 4, but overall batch size remains the same?` Didn't understand..
Answer: From one septuplet, which is called a "sample" here, there are 4 different triplets that are obtained (as explained in a previous answer). However, the losses from these 4 triplets are all averaged and the optimizer only steps every 4 triplets, making the ending batch size the same. I'm not sure if there could be side effects from doing this.

## Did the HDv3 weights work with IFNet_m?
- What other weights are available and differences? size/parameters?
- Answer: I tried a variety of other weights with IFNet_m a while ago, including HDv3, so far only the original model's works. I think the authors only used IFNet to generate the weights available online. To my knowledge, the format of the weights are the same.

# Suggestions and random thoughts
### Overfit Test
- To test the correctness of their code and your code 
- Training and testing on an extremely small dataset, check if very high training and testing accuracies are being achieved. 
- This is a simple to test and obvious indication of something being wrong.

### Motion Primitives
- What is the distribution of frame-to-frame motion in the training data (speed, rotation rates, etc.)?
- What is the distribution of frame-to-frame motion in the FPV footage (speed, rotation rates, etc.)?
- Typical optical flow magnitude (in pixels) and Viewpoint shift (rotation angle, parallax) are exploratory metrics. 
- Effect of Lens distortion (FPV cameras could have wide-angle compared to the dataset)
- Illumination changes (Indoors vs Outdoors) - distribution of Frame-to-frame change
- Standard RIFE skips nearly-identical frames (SSIM > 0.996) - similar for 10 FPS FPV?

If the two distributions are misaligned beyond a certain margin, RIFE architecture might be the limiting factor.


### Quality Metrics w.r.t. our application (Slide 28 in Emily's 10/17 presentation):
- PSNR/SSIM relate to perceptual quality
- Temporal smoothness (no judder/ghosting)
- Motion boundary sharpness - Optical Flow based metric
- Per-Texture performance
- Motion blur at source vs Performance
- Real-time performance vs. quality tradeoff - ties into project goal

- **This is helpful**: Some common artifacts pertaining to Video generation  

| Artifact | Description | Primary Causes in Video Generation  |
| ----- | ----- | ----- |
| Ghosting | Faint trails or blurry outlines following moving objects. | Occurs when data from previous frames is "smeared" or averaged with the current frame to smooth edges (e.g., Temporal Anti-Aliasing (TAA) in games), or due to slow pixel response times in displays. In AI, it results from motion prediction errors.  |
| Blur | Loss of fine detail, making parts or all of the video appear out of focus or "fuzzy". | Often caused by lossy compression which removes high-frequency image data, or by AI models struggling to render complex textures or scenes at high resolution.  |
| Temporal Bleeding | Similar to color bleeding but across time, where color or texture from one frame "bleeds" into the next inconsistently, often appearing as shimmering or flickering. | Stems from temporal instability in the generation model, where details are not consistently rendered across consecutive frames.  |
| Wrong Motion / Temporal Instability | Unnatural, jumpy, wobbly, or inconsistent movement of objects or the entire scene (flickering, stuttering, "texture floating"). | A significant issue in AI video, caused by models failing to maintain true film continuity across sequences, relying on imperfect frame interpolation, or misaligned motion vectors.  |

*I think our current issue is **temporal bleeding**?*  So we should keep an eye out for techniques to mitigate that..


### Distributed processing could be a future requirement



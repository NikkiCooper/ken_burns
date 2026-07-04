# **Bilateral Filter - Complete User Guide** 📖
_CUDA-Accelerated Video Enhancement for pyV Player_
## **📋 Table of Contents**
- [What is a Bilateral Filter?](#what-is-a-bilateral-filter)
- [Understanding Performance vs Video Frame Rates](#understanding-performance-vs-video-frame-rates)
- [Parameter Controls](#parameter-controls)
- [CUDA Performance Guide](#cuda-performance-guide)
- [Optimization Settings](#optimization-settings)
- [Hardware Recommendations](#hardware-recommendations)
- [Troubleshooting](#troubleshooting)
- [Quick Reference](#quick-reference)

## **🎯 What is a Bilateral Filter?**
A bilateral filter is an **advanced noise reduction technique** that **smooths images while preserving sharp edges**. Unlike simple blur filters that soften everything equally, bilateral filtering is "smart" - it:
✅ **Removes noise** from smooth areas (like skin, sky, walls)
✅ **Preserves details** at edges (like hair, text, object boundaries)
✅ **Maintains sharpness** where it matters most
✅ **Reduces video artifacts** from compression or low light
### **🎬 Perfect For:**
- **Old/vintage videos** with film grain or digital noise
- **Low-light footage** with visible noise
- **Compressed videos** with artifacts
- **Live streams** or webcam footage
- **Upscaled content** that looks pixelated

## **🚨 Understanding Performance vs Video Frame Rates**
### **⚠️ Important Clarification: Two Different "FPS" Concepts**
**📹 Video Content FPS (Your Video Files):**
- **25 FPS:** PAL standard (Europe/UK)
- **30 FPS:** NTSC standard (US/Canada)
- **60 FPS:** High frame rate content
- This is the **original recording rate** - it **never changes**

**🖥️ GPU Processing FPS (CUDA Performance):**
- How fast your **GPU can process** each frame through the filter
- **Completely separate** from your video's frame rate
- This determines if playback is **smooth or stuttering**

### **🔍 Real-World Example:**
``` 
Your 30 FPS video = 30 frames every second (fixed)
Your GPU processes at 90 FPS = Can handle 90 frames every second

Result: GPU is 3x faster than needed = Smooth playback with filtering
```

``` 
Your 30 FPS video = 30 frames every second (fixed)  
Your GPU processes at 20 FPS = Can only handle 20 frames every second

Result: GPU too slow = Stuttering, dropped frames, laggy playback
```
### **✅ The Goal:**
**Your GPU's processing speed should be at least 2x your video's frame rate for smooth playback.**
## **🎛️ Parameter Controls**
### **🔧 Diameter (3-15)**
**What it controls:** Size of the filtering neighborhood
- **3-5:** **Subtle smoothing**, very fast processing
- **7-9:** **Balanced filtering**, good performance _(Recommended)_
- **11-15:** **Strong smoothing**, slower processing

**⚠️ Performance Impact:** **Highest** - each increase roughly doubles processing time
### **🎨 Sigma Color (10-100)**
**What it controls:** How much color difference is allowed in smoothing
- **10-30:** **Conservative** - only very similar colors blend
- **40-70:** **Balanced** - natural-looking smoothing _(Recommended)_
- **80-100:** **Aggressive** - strong noise reduction, may look artificial

**⚠️ Performance Impact:** **Moderate** - higher values use more GPU memory
### **📐 Sigma Space (10-100)**
**What it controls:** How far the filter reaches spatially
- **10-30:** **Short reach** - only nearby pixels influence each other
- **40-70:** **Medium reach** - good balance _(Recommended)_
- **80-100:** **Long reach** - wide area smoothing

**⚠️ Performance Impact:** **Moderate** - affects GPU memory usage
### **💪 Intensity (0-100%)**
**What it controls:** Strength of the final effect
- **0%:** Filter disabled
- **30-50%:** **Subtle enhancement**
- **60-80%:** **Noticeable improvement** _(Recommended)_
- **90-100%:** **Maximum effect**

**⚠️ Performance Impact:** - adjust freely! **None**
## **🚀 CUDA Performance Guide**
### **🔥 CUDA Acceleration Benefits:**
- **CPU Processing:** ~5-15 FPS for HD video
- **CUDA Processing:** ~30-120 FPS for HD video
- **Performance Gain:** **6-25x faster** depending on your GPU

### **⚡ Real-time Processing Capabilities:**
- ✅ **4K Video:** Smooth playback with filtering
- ✅ **Multiple Effects:** Stack filters without lag
- ✅ **Live Preview:** Instant parameter adjustments
- ✅ **Background Processing:** Minimal CPU usage

## **⚙️ Optimization Settings**
### **🏎️ Performance Settings (Recommended Starting Point)**
_For smooth playback on most hardware:_
``` 
Diameter: 7          (Sweet spot for CUDA cores)
Sigma Color: 40      (Balanced GPU memory usage)  
Sigma Space: 40      (Optimal CUDA thread utilization)
Intensity: 70%       (Visible improvement, no performance cost)
```
**✅ Expected Results:**
- **30 FPS videos:** GPU processes at 60-120+ FPS = **Perfectly smooth**
- **25 FPS videos:** GPU processes at 50-100+ FPS = **Perfectly smooth**

### **⚖️ Balanced Settings**
_When Performance Settings run smoothly:_
``` 
Diameter: 9          (Better quality, still fast)
Sigma Color: 60      (Natural-looking results)
Sigma Space: 60      (Effective smoothing)  
Intensity: 70%       (Strong visible effect)
```
### **🎨 Maximum Quality Settings**
_For powerful GPUs or when quality matters most:_
``` 
Diameter: 11         (Highest quality, slower)
Sigma Color: 80      (Aggressive smoothing)
Sigma Space: 80      (Wide area processing)
Intensity: 85%       (Maximum effect)
```
### **🚨 Emergency Settings**
_When even Performance Settings are too slow:_
``` 
Diameter: 5          (Minimal processing)
Sigma Color: 25      (Light smoothing)
Sigma Space: 25      (Short reach)
Intensity: 50%       (Reduced effect)
```
## **🎮 Hardware Recommendations**
### **🏆 Excellent Performance (RTX 30/40 Series, GTX 16 Series)**
**RTX 4090/4080:**
- **30 FPS videos:** Any settings work perfectly
- **4K content:** Maximum Quality at 60+ processing FPS

**RTX 4070/3080:**
- **30 FPS videos:** Maximum Quality runs smoothly
- **4K content:** Balanced Settings recommended

**RTX 3070/4060:**
- **30 FPS videos:** Balanced Settings work great
- **4K content:** Performance Settings for smooth playback

### **✅ Good Performance (GTX 10 Series, RTX 20 Series)**
**RTX 2070/2080:**
- **30 FPS videos:** Balanced Settings
- **1440p content:** Performance Settings for 4K

**GTX 1070/1080:**
- **30 FPS videos:** Performance Settings work well
- **1080p content:** Can try Balanced Settings

### **⚠️ Basic Performance (Older Cards)**
**GTX 1060 6GB:**
- **30 FPS videos:** Performance Settings (may need Emergency Settings for 4K)
- **1080p content:** Stick to Performance Settings

**GTX 1050/1060 3GB:**
- **30 FPS videos:** Emergency Settings recommended
- Focus on **1080p content** or lower

## **📺 Video Resolution Guidelines**
### **📱 1080p Video (1920×1080):**
``` 
Any modern CUDA GPU: Start with Balanced Settings
Most GPUs will handle Maximum Quality smoothly
```
### **🖥️ 1440p Video (2560×1440):**
``` 
RTX 30/40 series: Balanced to Maximum Quality
RTX 20/GTX 16: Performance to Balanced Settings
GTX 10 series: Performance Settings  
```
### **📺 4K Video (3840×2160):**
``` 
RTX 4080/4090: Balanced to Maximum Quality
RTX 3070+/4060+: Performance to Balanced Settings  
RTX 3060/2070: Performance Settings
Older cards: Emergency Settings or consider downscaling
```
## **🔧 Troubleshooting**
### **🐌 If Video Stutters or Drops Frames:**
1. **Lower the Diameter first** (biggest performance impact)
    - Try: 11 → 9 → 7 → 5

2. **Reduce Sigma values**
    - Try: 80 → 60 → 40 → 25

3. **Check your GPU usage** (should be 70-95% during filtering)
4. **Ensure adequate VRAM** (4GB+ recommended for 4K)

### **💾 Memory Issues (Crashes, Freezing):**
1. **Lower all parameters significantly**
2. **Check available VRAM** with `nvidia-smi`
3. **Close other GPU-intensive applications**
4. **Consider your hardware limitations**

### **🔄 If Changes Don't Apply:**
1. **Adjust Intensity** to see immediate effect
2. **Try toggling filter on/off** to verify it's working
3. **Restart the application** if settings seem stuck

## **📋 Quick Reference**
### **🚀 Quick Start Guide:**
Start with **Performance Settings** **Step 1:**
``` 
Diameter: 7, Sigma Color: 40, Sigma Space: 40, Intensity: 70%
```
Test with your typical video content **Step 2:**
- Does your **30 FPS video play smoothly?** ✅ Try Balanced Settings
- Does it **stutter or drop frames?** ⚠️ Try Emergency Settings

Fine-tune based on results **Step 3:**
- **Smooth playback?** → Gradually increase parameters
- **Any stuttering?** → Reduce Diameter first, then Sigma values

### **⚡ Performance Expectations:**

| GPU Class | 30 FPS Video | Performance Settings | Balanced Settings | Max Quality |
| --- | --- | --- | --- | --- |
| RTX 4080+ | ✅ Perfect | ✅ Perfect | ✅ Perfect | ✅ Perfect |
| RTX 3070+ | ✅ Perfect | ✅ Perfect | ✅ Good | ⚠️ May stutter |
| RTX 2070+ | ✅ Perfect | ✅ Good | ⚠️ May stutter | ❌ Too slow |
| GTX 1060+ | ✅ Good | ⚠️ May stutter | ❌ Too slow | ❌ Too slow |
### **🎯 Content-Specific Recommendations:**
**Animation/CGI:** Lower settings work fine (clean source) **Live Action:** Standard Performance Settings
**Old/Noisy Footage:** May need Balanced Settings despite performance cost **Streaming/Webcam:** Performance Settings for real-time processing
## **🏁 Summary**
**The bilateral filter transforms your video viewing experience by removing noise while preserving important details.** With CUDA acceleration, you get **professional-quality filtering in real-time.**
**Remember:** Your videos will always play at their original frame rate (25/30 FPS). The settings control whether your GPU can **process those frames fast enough** for smooth playback.
**Start conservative, increase gradually, and find your hardware's sweet spot for the perfect balance of quality and performance!** 🎥✨

# 🏆 Practical Performance Enhancement of CNN Core Execution Using Pthread, OpenMP, and CUDA  

## 📌 Introduction  

Based on the core convolution calculations of the commonly used **Convolutional Neural Network (CNN)** in artificial intelligence, this project explores **performance enhancement (speedup) using multi-threading** architectures such as **Pthread, OpenMP, and CUDA**.  

🔹 **Pthread** and **OpenMP** are executed on a personal computer.  
🔹 **CUDA** is executed on **Google Colab** to analyze performance in large-scale multi-threaded environments.  

## 🛠 1. Preprocessing  

With rapid advancements in **electronic technology**, computing power has significantly improved. Many **AI models that struggled due to computational limitations** have been greatly enhanced, enabling rapid AI development.  

### 🔍 **Why CNN?**  
CNN is a widely used AI architecture, where **convolution operations form its core**. The convolution computation follows the formula shown in **Figure 1**, which serves as the basis for **evaluating processing efficiency** and **multi-threaded performance enhancements**.  

### 🎨 **Filters Used**  
Different filters serve different image processing needs. In this study, we apply:  

✅ **Sobel Filter** → Detects **edges** (Figure 3)  
✅ **Gaussian Filter** → Performs **blurring** (Figure 4)  

Since convolution requires **extensive matrix computations**, substantial computational power is needed. **Multi-threading** allows multiple tasks to be executed in parallel, significantly increasing speed. More **threads** → **less execution time** → **higher speedup** 🚀.  

## ⚙️ 2. Implementation with Pthread  

The first implementation uses **Pthread**. It employs:  

🔹 `pthread_create` → To create multiple threads  
🔹 `pthread_join` → To merge the execution results  

### ⚠️ **Challenges Faced**  
Initially, we assumed that simply adding these functions would enable multi-threaded execution. However, **incorrect results** appeared. Further debugging revealed issues with:  

✅ **Upper and lower bounds** → Threads didn’t evenly divide pixel counts  
✅ **Memory allocation adjustments**  

After fixing these, **Pthread implementation ran correctly** 🎉.  

### 📊 **Performance Evaluation**  
The program processed images of **various resolutions (320, 1280, 2880, 6000, 11400)** using **Sobel and Gaussian filters**. **Speedup (efficiency gain)** was calculated as the ratio of **single-threaded to multi-threaded execution time** (**Table 1**).  

🔹 **More threads generally improve performance** but not always **linearly** 📈  
🔹 **Gaussian filtering** achieves **higher speedup** than **Sobel filtering**  
🔹 **Some cases showed speedup exceeding thread count** → due to **overhead reduction**  

## 🚀 3. Implementation with OpenMP  

Next, we switch to **OpenMP**, which is **simpler** and **automatically handles boundary conditions** compared to Pthread.  

### ✅ **Why OpenMP?**  
🔹 **Easy syntax**  
🔹 **No need to manually handle thread synchronization**  
🔹 **Ideal for parallel processing**  

A snippet of **core OpenMP implementation** is shown in **Figure 6**.  

### 📊 **Performance Results**  
The same experiment was conducted using **OpenMP**, and **Table 2** summarizes **speedup results across different thread counts**. Overall, OpenMP achieved **excellent performance gains** 🎯.  

## ⚡ 4. CUDA and Google Colab Implementation  

To further analyze performance gains, **CUDA implementation** was executed on **Google Colab**.  

### 🔥 **GPU Used: NVIDIA T4** (Figure 7)  
✅ **2560 CUDA cores** → Much more powerful than Jetson Nano  
✅ **Large-scale thread parallelism**  

### ⚠️ **Challenges with CUDA**  
Unlike **Pthread and OpenMP**, CUDA requires:  
🔹 **Explicit memory allocation**  
🔹 **Data copying between CPU and GPU**  
🔹 **Memory deallocation**  

A portion of the **core CUDA implementation** is shown in **Figure 8**.  

### 📊 **Speedup Results**  
We tested CUDA using thread counts: **128, 256, 512, and 1024** and compared execution time between **CPU and GPU**. **Table 3** summarizes the results:  

🔥 **Sobel Filtering** → Speedup **150× to 504×**  
🔥 **Gaussian Filtering** → Speedup **198× to 917×**  

🔹 **Figure 9** shows a clear **trend of increasing performance** as **thread count increases** 📈🚀.  

## 🎯 Conclusion  

After extensive effort, the **multi-threaded CNN core implementation was successfully developed**. While not fully optimized, this project effectively demonstrates **parallel computing performance enhancements**.  

🎯 **Future Applications**  
With this strong foundation, we can further explore:  
✅ **Private blockchain implementation**  
✅ **Smart contract deployment**  
✅ **Other AI-related parallel computing applications**  

💡 **This project was both challenging and rewarding!** 🚀  

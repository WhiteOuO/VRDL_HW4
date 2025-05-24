# NYCU VRDL HW4 Image_Restoration by PromptIR
StudentID:110550128  
Name:蔡耀霆
## Introduction:
The task of this project is image restoration. The training and testing data have been specially designed, with noise uniformly distributed across the entire image. Each image contains only one type of noise, and there are only two types of noise: rain and snow. The deep learning model used in this project is based on the concept of PromptIR. The core objective of the model is to handle multiple types of noise within a unified model architecture. It introduces the concept of "prompts," where each prompt corresponds to a specific noise removal task. During the learning process, the model becomes increasingly accurate in interpreting these prompts, enabling better restoration of images affected by specific types of noise.
The model architecture in this project is built upon the framework provided in this GitHub repository: https://github.com/va1shn9v/PromptIR  
## Environment:  
你可以參照原作者的github建議的環境建議方式  
## Image Restoration Results

Below are examples of restored images from the PromptIR model:

<table>
  <tr>
    <td><img src="Restored_image/0.jpg" width="256" height="256" alt="Restored Rain Image 1"></td>
    <td><img src="degraded/0.jpg" width="256" height="256" alt="Restored Snow Image 1"></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="images/restored_rain2.jpg" width="256" height="256" alt="Restored Rain Image 2"></td>
    <td><img src="images/restored_snow2.jpg" width="256" height="256" alt="Restored Snow Image 2"></td>
  </tr>
</table>


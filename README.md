# PersoBench
### Benchmarking Personalized Response Generation in Large Language Models
![Python](https://img.shields.io/badge/Python-Compatible-green.svg)


## Abstract
While large language models (LLMs) have exhibited impressive conversational capabilities, their proficiency in delivering personalized responses remains unclear. Although recent benchmarks automatically evaluate persona consistency in role-playing contexts using LLM-based judgment, the evaluation of personalization in response generation remains underexplored. To address this gap, we present a new automated benchmarking framework, **PersoBench**, to evaluate the personalization ability of LLMs in persona-aware dialogue generation within a zero-shot setting. Our framework employs a structured pipeline comprising speaker-aware annotation, task-specific and context-driven prompt construction, response post-processing, and automated evaluation across multiple dimensions of generation quality. We assess the performance of three open-source and three closed-source LLMs using well-known datasets and a range of explicit metrics. Our analysis, conducted on three well-known persona-aware datasets, evaluates multiple dimensions of response quality, including fluency, diversity, coherence, and personalization, across both standard and chain-of-thought prompting methods. Our findings reveal that while LLMs excel at generating fluent and diverse responses, they are far from satisfactory in delivering personalized and coherent responses, considering both the conversation context and the provided personas.


## Figures

| ![First Image Description](Figures/PersoBench_Overview.jpg) | 
|:----------------------------------------------------------:|
| Fig.1 - Overview of the PersoBench automatic personalization benchmark framework.|

<div align="center" style="margin: 30px 0;">

<table>
  <tr>
    <td align="center">
      <img src="Figures/radar_vanilla_open_source.png" alt="open-source vanilla" width="300"><br/>
      (a)
    </td>
    <td align="center">
      <img src="Figures/radar_vanilla_closed_source.png" alt="closed-source vanilla" width="300"><br/>
      (b)
    </td>
    <td align="center">
      <img src="Figures/radar_cot_open_source.png" alt="open-source cot" width="300"><br/>
      (c)
    </td>
    <td align="center">
      <img src="Figures/radar_cot_closed_source.png" alt="closed-source cot" width="300"><br/>
      (d)
    </td>
  </tr>
</table>

<br/>

<em>
Performance analysis of (a) open-source LLMs in vanilla setting,  
(b) closed-source LLMs in vanilla setting,  
(c) open-source LLMs in COT setting,  
(d) closed-source LLMs in COT setting on the FoCus dataset.
</em>

</div>


<div align="center" style="margin: 30px 0;">

<table>
  <tr>
    <td align="center">
      <img src="Figures/focus_vanilla_setup_grouped_plus_scaled.png" alt="vanilla setup" width="350"><br/>
      (a) Vanilla setup
    </td>
    <td align="center">
      <img src="Figures/focus_cot_setup_grouped_plus_scaled.png" alt="cot setup" width="350"><br/>
      (b) COT setup
    </td>
  </tr>
</table>

<br/>

<em>
Analysis of response time, failure ratio, and engagingness of LLMs on the FoCus dataset:  
(a) Vanilla and (b) COT setups.
</em>

</div>



## 🚀 Quick Start

1. Clone the relevant repositories:
   ```bash
   git clone https://github.com/salehafzoon/PersoBench


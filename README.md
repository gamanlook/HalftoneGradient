# Halftone Gradient

<img width="1200" height="720" alt="halftone-gradient" src="https://github.com/user-attachments/assets/05c4e3a3-2cf7-4604-8a38-8f4c52ad159f" />


<br>

👉 [Live Demo](https://halftone-gradient.vercel.app/)

<br>

Paper Shaders offers many stunning WebGL effects, but it doesn't include an option that combines a dynamic Mesh Gradient with a Halftone filter. So I tried stitching these two effects together, and the result looks awesome! This is a dual-pass shader where the halftone dots flow dynamically with the gradient. Feel free to try it out!

Paper Shaders 提供了許多令人驚豔的WebGL特效，但裡面沒有「動態網格漸層(Mesh Gradient)」與「半色調網點(Halftone)」結合的選項。因此我嘗試將這兩種效果拼接在一起，效果很讚！這是一個能讓網點隨著漸層流動的雙層Shader，歡迎大家體驗看看！

<sub>Shaders by [Paper Shaders](https://shaders.paper.design) under [PolyForm Shield 1.0.0](https://polyformproject.org/licenses/shield/1.0.0).<br>Required Notice: Copyright Lost Coast Labs, Inc. (http://paper.design)</sub>




<br>
<br>

## 更新記錄（Changelog）

`v.1.1.1` `2026/04/30（四）` **🔧泡泡尖點修正**
- 修正了「Blobs泡泡模式」兩球相交時中心會出現突兀尖點的問題
  - 現在泡泡在融合與柔邊時的漸層效果像液體般圓滑自然

`v.1.1.0` `2026/04/29（三）` **🫧追加泡泡模式**
- 提供 Aurora / Blobs 選項可以做切換
  - Aurora是原本就有的，Blobs是追加的
  - Aurora(極光)：反距離加權 (Inverse Distance Weighting)，計算像素距離中心點多遠，距離越遠權重越低（衰減），完全沒有實體邊界
  - Blobs(泡泡)：融球 (Metaballs)，把多個 SDF 的數值用非線性的方式「加總」，超過某個閾值就填色，造就液體沾黏感
  - Blobs能做出類似ChatGPT生圖時的loading動畫，如下
<br>
<img width="600" height="360" alt="halftone-gradient ChatGPT-like Loading Style" src="https://github.com/user-attachments/assets/36edd7d9-e102-485e-91ac-323b3e767a04" />
<br>
<br>

`v.1.0.1` `2026/04/28（二）` **🔘調整滑桿元件**
- 原本滑桿太細，加粗了
- 製作客製版的選色器，不然不同裝置的原生選色器都很難用
- 彩虹漸層有將誤差偏移修正，圓球確實會停在對應的色相上

`v.1.0.0` `2026/04/27（一）` **🌌開始製作shader組合**
- 主要技術來源於 Paper Shader，發現可以自己重組出官方沒有提供的漂亮效果
